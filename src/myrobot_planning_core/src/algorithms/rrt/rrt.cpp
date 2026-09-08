#include "myrobot_planning_core/algorithms/rrt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "../common/goal_set_utils.hpp"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"

namespace fairino_planning {
namespace {

double jointDistance(const JointConfig& a, const JointConfig& b) { return (a - b).norm(); }

bool isFiniteConfig(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) return false;
    }
    return true;
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

RRT::RRT() : rng_(42) {}

PlanResult RRT::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    return planOnce(
        request.q_start,
        request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
        request.random_seed,
        &request);
}

PlanResult RRT::plan(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obs_origin;
    (void)obs_size;
    return planOnce(q_start, {q_goal}, 0);
}

PlanResult RRT::planOnce(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    unsigned int request_seed,
    const PlanRequestCore* request) {
    const auto started = std::chrono::steady_clock::now();
    PlanResult result;
    SearchRuntime runtime(request, started);
    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0U ? 42U : request_seed));

    const auto finish = [&](bool success, PlanningFailureCode code, const std::string& message) {
        result.success = success;
        result.failure_code = code;
        result.message = message;
        result.planning_time = runtime.elapsedSeconds();
        runtime.capture(result, success, result.path_cost, result.num_nodes);
        runtime.markStopReason(result, result.iterations >= params_.max_iterations);
        return result;
    };

    if (!collision_) {
        return finish(false, PlanningFailureCode::kInvalidInput, "RRT: null collision checker.");
    }
    if (goal_candidates.empty() || !isFiniteConfig(q_start) || !limits_.isWithin(q_start)) {
        return finish(false, PlanningFailureCode::kInvalidInput,
                      "RRT: non-finite or out-of-limit request state.");
    }
    if (!collision_->isStateValid(q_start)) {
        return finish(false, PlanningFailureCode::kGoalNotReached,
                      "RRT: start configuration in collision.");
    }
    for (const auto& q_goal : goal_candidates) {
        if (!isFiniteConfig(q_goal) || !limits_.isWithin(q_goal)) {
            return finish(false, PlanningFailureCode::kInvalidInput,
                          "RRT: goal configuration is non-finite or out of joint limits.");
        }
        if (!collision_->isStateValid(q_goal)) {
            return finish(false, PlanningFailureCode::kGoalNotReached,
                          "RRT: goal configuration in collision.");
        }
    }

    const double validation_distance = std::max(1e-6, params_.validation_distance);
    const auto direct_path = goal_set::shortestStrictDirectPath(
        *collision_, q_start, goal_candidates, validation_distance);
    if (direct_path) {
        result.path = direct_path->path;
        result.path_cost = direct_path->cost;
        result.num_nodes = 2;
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = result.path_cost;
        return finish(true, PlanningFailureCode::kNone, "");
    }

    RRTTree tree(params_.max_iterations + static_cast<int>(goal_candidates.size()) + 2);
    tree.addNode(q_start, -1, 0.0);
    int best_goal = -1;
    std::vector<JointConfig> best_path = direct_path ? direct_path->path : std::vector<JointConfig>{};
    double best_cost = direct_path ? direct_path->cost : std::numeric_limits<double>::infinity();

    for (int iteration = 1; iteration <= params_.max_iterations; ++iteration) {
        if (runtime.shouldStop()) break;
        result.iterations = iteration;
        runtime.capture(result, best_goal >= 0, best_cost, tree.size());
        ++result.sample_attempts;
        JointConfig q_sample = limits_.sampleUniform(rng_);
        if (params_.goal_bias > 0.0) {
            std::uniform_real_distribution<double> coin(0.0, 1.0);
            if (coin(rng_) < params_.goal_bias) {
                std::uniform_int_distribution<size_t> pick(0U, goal_candidates.size() - 1U);
                q_sample = goal_candidates[pick(rng_)];
            }
        }
        const int nearest = tree.nearest(q_sample);
        const JointConfig q_near = tree.node(nearest).state;
        const JointConfig q_new = limits_.clamp(steer(q_near, q_sample, params_.max_step));
        if (!collision_->isStateValid(q_new) ||
            !collision_->isMotionValid(q_near, q_new, validation_distance)) {
            continue;
        }

        const double new_cost = tree.node(nearest).cost + jointDistance(q_near, q_new);
        const int new_index = tree.addNode(q_new, nearest, new_cost);
        ++result.accepted_samples;

        const auto nearest_goal = std::min_element(
            goal_candidates.begin(), goal_candidates.end(),
            [&](const JointConfig& a, const JointConfig& b) {
                return jointDistance(q_new, a) < jointDistance(q_new, b);
            });
        const double goal_distance = jointDistance(q_new, *nearest_goal);
        if (goal_distance >= params_.goal_threshold ||
            !collision_->isMotionValid(q_new, *nearest_goal, validation_distance)) {
            continue;
        }

        const double candidate_cost = new_cost + goal_distance;
        if (candidate_cost + 1e-12 >= best_cost) continue;
        const int goal_index = tree.addNode(*nearest_goal, new_index, candidate_cost);
        const auto candidate_path = tree.backtrack(goal_index);
        if (!strictlyValidPath(*collision_, candidate_path, validation_distance)) continue;
        best_goal = goal_index;
        best_cost = candidate_cost;
        best_path = candidate_path;
        result.path = best_path;
        result.path_cost = best_cost;
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = best_cost;
        break;
    }

    if (best_goal < 0 && best_path.empty()) {
        return finish(false, runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached,
            runtime.deadlineReached() ? "deadline_no_solution" : "RRT failed: goal not reached.");
    }
    result.path = best_goal >= 0 ? tree.backtrack(best_goal) : best_path;
    result.path_cost = best_cost;
    result.num_nodes = tree.size();
    return finish(true, PlanningFailureCode::kNone, "");
}

}  // namespace fairino_planning
