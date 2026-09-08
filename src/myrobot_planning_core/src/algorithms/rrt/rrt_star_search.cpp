#include "rrt_star_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "myrobot_planning_core/tree/rrt_tree.h"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"

namespace fairino_planning {
namespace {

double jointDistance(const JointConfig& a, const JointConfig& b) { return (a - b).norm(); }
double jointDistanceSq(const JointConfig& a, const JointConfig& b) { return (a - b).squaredNorm(); }

bool isFiniteConfig(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) return false;
    }
    return true;
}

JointConfig steerBounded(const JointConfig& from, const JointConfig& to, double max_step) {
    const JointConfig delta = to - from;
    const double distance = delta.norm();
    if (distance < 1e-12) return from;
    return from + delta * (std::min(max_step, distance) / distance);
}

double rewireRadius(const PlanningParams& params, int node_count) {
    const double count = static_cast<double>(std::max(node_count, 2));
    const double radius = params.gamma * std::pow(std::log(count) / count, 1.0 / NUM_JOINTS);
    return std::min(params.max_rewire_radius, std::max(radius, params.max_step * 1.2));
}

bool strictlyValidPath(
    const CollisionInterface& collision,
    const std::vector<JointConfig>& path,
    double validation_distance) {
    if (path.empty() || !collision.isStateValid(path.front())) return false;
    for (size_t i = 1; i < path.size(); ++i) {
        if (!collision.isStateValid(path[i]) ||
            !collision.isMotionValid(path[i - 1U], path[i], validation_distance)) {
            return false;
        }
    }
    return true;
}

}  // namespace

PlanResult runRrtStarSearch(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    RrtStarSampler sampler,
    const std::string& planner_name,
    const PlanRequestCore* request) {
    const auto started = std::chrono::steady_clock::now();
    PlanResult result;
    SearchRuntime runtime(request, started);
    const auto finish = [&](bool success, PlanningFailureCode code, const std::string& message) {
        result.success = success;
        result.failure_code = code;
        result.message = message;
        result.planning_time = runtime.elapsedSeconds();
        runtime.capture(result, success, result.path_cost, result.num_nodes);
        runtime.markStopReason(result, result.iterations >= params.max_iterations);
        return result;
    };

    if (goal_candidates.empty() || !isFiniteConfig(q_start) || !limits.isWithin(q_start)) {
        return finish(false, PlanningFailureCode::kInvalidInput,
                      planner_name + ": non-finite or out-of-limit request state.");
    }
    if (!collision.isStateValid(q_start)) {
        return finish(false, PlanningFailureCode::kGoalNotReached,
                      planner_name + ": start configuration in collision.");
    }
    for (const auto& q_goal : goal_candidates) {
        if (!isFiniteConfig(q_goal) || !limits.isWithin(q_goal)) {
            return finish(false, PlanningFailureCode::kInvalidInput,
                          planner_name + ": goal configuration is non-finite or out of joint limits.");
        }
        if (!collision.isStateValid(q_goal)) {
            return finish(false, PlanningFailureCode::kGoalNotReached,
                          planner_name + ": goal configuration in collision.");
        }
    }

    const double validation_distance = std::max(1e-6, params.validation_distance);
    RRTTree tree(params.max_iterations + static_cast<int>(goal_candidates.size()) + 2);
    tree.addNode(q_start, -1, 0.0);

    int best_goal_index = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int first_goal_sample_attempt = -1;
    const int post_solution_budget = std::max(0, params.post_solution_sample_attempts);
    for (size_t goal_index = 0; goal_index < goal_candidates.size(); ++goal_index) {
        const JointConfig& goal = goal_candidates[goal_index];
        const double direct_cost = jointDistance(q_start, goal);
        if (direct_cost + 1e-12 >= best_cost ||
            !collision.isMotionValid(q_start, goal, validation_distance)) {
            continue;
        }
        best_goal_index = tree.addNode(goal, 0, direct_cost);
        best_cost = direct_cost;
    }
    if (best_goal_index >= 0) {
        first_goal_sample_attempt = 0;
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = best_cost;
    }

    for (int iteration = 1; iteration <= params.max_iterations; ++iteration) {
        if (runtime.shouldStop()) break;
        if (post_solution_budget > 0 && first_goal_sample_attempt >= 0 &&
            result.sample_attempts - first_goal_sample_attempt >= post_solution_budget) {
            result.effort_stats.post_solution_budget_complete = true;
            break;
        }
        result.iterations = iteration;
        if (best_goal_index >= 0) best_cost = tree.node(best_goal_index).cost;
        runtime.capture(result, best_goal_index >= 0, best_cost, tree.size());

        ++result.sample_attempts;
        const RrtStarSearchState search_state{
            best_goal_index >= 0, best_goal_index >= 0 ? best_cost : 0.0};
        const JointConfig q_rand = sampler(search_state);
        if (!isFiniteConfig(q_rand)) continue;
        const int nearest_index = tree.nearest(q_rand);
        const JointConfig& q_near = tree.node(nearest_index).state;
        const JointConfig q_new = limits.clamp(steerBounded(q_near, q_rand, params.max_step));
        if (!collision.isStateValid(q_new)) continue;

        auto near_set = tree.nearRadius(q_new, rewireRadius(params, tree.size()));
        if (near_set.empty()) near_set.push_back(nearest_index);
        if (static_cast<int>(near_set.size()) > params.max_near) {
            std::partial_sort(
                near_set.begin(), near_set.begin() + params.max_near, near_set.end(),
                [&](int lhs, int rhs) {
                    return jointDistanceSq(tree.node(lhs).state, q_new) <
                           jointDistanceSq(tree.node(rhs).state, q_new);
                });
            near_set.resize(params.max_near);
        }

        int best_parent = -1;
        double best_new_cost = std::numeric_limits<double>::infinity();
        for (const int candidate_parent : near_set) {
            const double candidate_cost = tree.node(candidate_parent).cost +
                jointDistance(tree.node(candidate_parent).state, q_new);
            if (candidate_cost >= best_new_cost ||
                !collision.isMotionValid(tree.node(candidate_parent).state, q_new, validation_distance)) {
                continue;
            }
            best_parent = candidate_parent;
            best_new_cost = candidate_cost;
        }
        if (best_parent < 0) continue;

        const int new_index = tree.addNode(q_new, best_parent, best_new_cost);
        ++result.accepted_samples;

        if (params.rewire_every_k <= 0 || iteration % params.rewire_every_k == 0) {
            const int rewire_limit = std::min(params.rewire_max_neighbors, static_cast<int>(near_set.size()));
            for (int i = 0; i < rewire_limit; ++i) {
                const int candidate_child = near_set[i];
                if (candidate_child == best_parent || candidate_child == new_index) continue;
                const double candidate_cost = tree.node(new_index).cost +
                    jointDistance(q_new, tree.node(candidate_child).state);
                if (candidate_cost + 1e-12 >= tree.node(candidate_child).cost ||
                    !collision.isMotionValid(q_new, tree.node(candidate_child).state, validation_distance)) {
                    continue;
                }
                tree.reparent(candidate_child, new_index, candidate_cost);
            }
        }

        const auto nearest_goal = std::min_element(
            goal_candidates.begin(), goal_candidates.end(),
            [&](const JointConfig& lhs, const JointConfig& rhs) {
                return jointDistance(q_new, lhs) < jointDistance(q_new, rhs);
            });
        const double goal_distance = jointDistance(q_new, *nearest_goal);
        if (goal_distance >= params.goal_threshold ||
            !collision.isMotionValid(q_new, *nearest_goal, validation_distance)) {
            continue;
        }

        const double candidate_cost = best_new_cost + goal_distance;
        if (candidate_cost + 1e-12 >= best_cost) continue;
        const int goal_index = tree.addNode(*nearest_goal, new_index, candidate_cost);
        const auto candidate_path = tree.backtrack(goal_index);
        if (!strictlyValidPath(collision, candidate_path, validation_distance)) continue;

        best_goal_index = goal_index;
        best_cost = candidate_cost;
        if (first_goal_sample_attempt < 0) {
            first_goal_sample_attempt = result.sample_attempts;
            result.first_solution_time_s = runtime.elapsedSeconds();
            result.first_solution_path_cost = best_cost;
        }
    }

    if (first_goal_sample_attempt >= 0) {
        result.effort_stats.post_solution_sample_attempts = std::max(
            0, result.sample_attempts - first_goal_sample_attempt);
    }

    if (best_goal_index < 0) {
        return finish(false, runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached,
                      runtime.deadlineReached() ? "deadline_no_solution" : planner_name + " failed: goal not reached.");
    }

    result.path = tree.backtrack(best_goal_index);
    if (!strictlyValidPath(collision, result.path, validation_distance)) {
        return finish(false, PlanningFailureCode::kCollision,
                      planner_name + " final path invalid.");
    }
    result.path_cost = tree.node(best_goal_index).cost;
    result.num_nodes = tree.size();
        return finish(true, PlanningFailureCode::kNone, "");
}

}  // namespace fairino_planning
