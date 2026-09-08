#include "myrobot_planning_core/algorithms/prm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>

#include "../common/goal_set_utils.hpp"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"
#include <utility>

namespace fairino_planning {
namespace {

struct RoadmapNode {
    JointConfig state;
    std::vector<std::pair<int, double>> edges;
};

double jointDistance(const JointConfig& lhs, const JointConfig& rhs) { return (lhs - rhs).norm(); }

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

std::vector<int> nearestIndices(
    const std::vector<RoadmapNode>& roadmap,
    const JointConfig& query,
    int count) {
    std::vector<int> indices(roadmap.size());
    for (size_t index = 0; index < roadmap.size(); ++index) indices[index] = static_cast<int>(index);
    const size_t keep = std::min(indices.size(), static_cast<size_t>(std::max(0, count)));
    std::partial_sort(
        indices.begin(), indices.begin() + keep, indices.end(),
        [&](int lhs, int rhs) {
            return jointDistance(roadmap[lhs].state, query) < jointDistance(roadmap[rhs].state, query);
        });
    indices.resize(keep);
    return indices;
}

}  // namespace

PRM::PRM() : rng_(42) {}

PlanResult PRM::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    return planOnce(
        request.q_start,
        request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
        request.random_seed, &request);
}

PlanResult PRM::plan(
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
    return planOnce(q_start, {q_goal}, 0U);
}

PlanResult PRM::planOnce(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    unsigned int request_seed,
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
        runtime.markStopReason(result, result.iterations >= params_.max_iterations);
        return result;
    };
    if (!collision_) {
        return finish(false, PlanningFailureCode::kInvalidInput, "PRM: null collision checker.");
    }
    if (goal_candidates.empty() || !isFiniteConfig(q_start) || !limits_.isWithin(q_start)) {
        return finish(false, PlanningFailureCode::kInvalidInput,
                      "PRM: non-finite or out-of-limit request state.");
    }
    if (!collision_->isStateValid(q_start)) {
        return finish(false, PlanningFailureCode::kGoalNotReached, "PRM: start configuration in collision.");
    }
    for (const auto& goal : goal_candidates) {
        if (!isFiniteConfig(goal) || !limits_.isWithin(goal)) {
            return finish(false, PlanningFailureCode::kInvalidInput,
                          "PRM: goal configuration is non-finite or out of joint limits.");
        }
        if (!collision_->isStateValid(goal)) {
            return finish(false, PlanningFailureCode::kGoalNotReached, "PRM: goal configuration in collision.");
        }
    }

    const double validation_distance = std::max(1e-6, params_.validation_distance);
    const auto direct_path = goal_set::shortestStrictDirectPath(
        *collision_, q_start, goal_candidates, validation_distance);

    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0U ? 42U : request_seed));
    const int k_neighbors = std::max(1, params_.prm.k_neighbors);
    std::vector<RoadmapNode> roadmap;
    roadmap.reserve(params_.max_iterations);
    std::vector<JointConfig> best_path = direct_path ? direct_path->path : std::vector<JointConfig>{};
    double best_cost = direct_path ? direct_path->cost : std::numeric_limits<double>::infinity();
    int first_solution_sample_attempt = direct_path ? 0 : -1;
    const int post_solution_budget = std::max(0, params_.post_solution_sample_attempts);
    int query_count = 0;
    if (direct_path) {
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = best_cost;
    }

    const auto queryRoadmap = [&]() -> bool {
        if (roadmap.empty()) return false;
        const int start_index = static_cast<int>(roadmap.size());
        const int first_goal_index = start_index + 1;
        std::vector<std::vector<std::pair<int, double>>> graph(roadmap.size() + 1U + goal_candidates.size());
        for (size_t index = 0; index < roadmap.size(); ++index) graph[index] = roadmap[index].edges;

        const auto connectQuery = [&](const JointConfig& query, int query_index) {
            for (const int node_index : nearestIndices(roadmap, query, k_neighbors)) {
                const double cost = jointDistance(query, roadmap[node_index].state);
                if (!collision_->isMotionValid(query, roadmap[node_index].state, validation_distance)) continue;
                graph[query_index].push_back({node_index, cost});
                graph[node_index].push_back({query_index, cost});
            }
        };
        connectQuery(q_start, start_index);
        for (size_t index = 0; index < goal_candidates.size(); ++index) {
            connectQuery(goal_candidates[index], first_goal_index + static_cast<int>(index));
        }
        if (graph[start_index].empty()) return false;

        ++query_count;
        const double infinity = std::numeric_limits<double>::infinity();
        std::vector<double> distances(graph.size(), infinity);
        std::vector<int> parents(graph.size(), -1);
        using QueueValue = std::pair<double, int>;
        std::priority_queue<QueueValue, std::vector<QueueValue>, std::greater<QueueValue>> queue;
        distances[start_index] = 0.0;
        queue.push({0.0, start_index});
        while (!queue.empty()) {
            const auto [distance, node] = queue.top();
            queue.pop();
            if (distance != distances[node]) continue;
            for (const auto& [neighbor, edge_cost] : graph[node]) {
                const double candidate = distance + edge_cost;
                if (candidate + 1e-12 >= distances[neighbor]) continue;
                distances[neighbor] = candidate;
                parents[neighbor] = node;
                queue.push({candidate, neighbor});
            }
        }

        int best_goal = -1;
        double query_best_cost = std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < goal_candidates.size(); ++index) {
            const int goal_index = first_goal_index + static_cast<int>(index);
            if (distances[goal_index] < query_best_cost) {
                best_goal = goal_index;
                query_best_cost = distances[goal_index];
            }
        }
        if (best_goal < 0 || !std::isfinite(query_best_cost) || query_best_cost + 1e-12 >= best_cost) {
            return false;
        }

        std::vector<JointConfig> candidate;
        for (int node = best_goal; node >= 0; node = parents[node]) {
            if (node == start_index) candidate.push_back(q_start);
            else if (node >= first_goal_index) candidate.push_back(goal_candidates[node - first_goal_index]);
            else candidate.push_back(roadmap[node].state);
        }
        std::reverse(candidate.begin(), candidate.end());
        if (!strictlyValidPath(*collision_, candidate, validation_distance)) return false;
        best_cost = query_best_cost;
        best_path = std::move(candidate);
        return true;
    };

    for (int iteration = 1; iteration <= params_.max_iterations; ++iteration) {
        if (runtime.shouldStop()) break;
        if (post_solution_budget > 0 && first_solution_sample_attempt >= 0 &&
            result.sample_attempts - first_solution_sample_attempt >= post_solution_budget) {
            result.effort_stats.post_solution_budget_complete = true;
            break;
        }
        result.iterations = iteration;
        runtime.capture(result, !best_path.empty(), best_cost, static_cast<int>(roadmap.size()));

        ++result.sample_attempts;
        const JointConfig sample = limits_.sampleUniform(rng_);
        if (!collision_->isStateValid(sample)) continue;
        const int new_index = static_cast<int>(roadmap.size());
        roadmap.push_back({sample, {}});
        ++result.accepted_samples;
        for (const int neighbor : nearestIndices(roadmap, sample, k_neighbors + 1)) {
            if (neighbor == new_index ||
                !collision_->isMotionValid(sample, roadmap[neighbor].state, validation_distance)) {
                continue;
            }
            const double cost = jointDistance(sample, roadmap[neighbor].state);
            roadmap[new_index].edges.push_back({neighbor, cost});
            roadmap[neighbor].edges.push_back({new_index, cost});
        }

        const bool query_due = static_cast<int>(roadmap.size()) >= k_neighbors &&
            (static_cast<int>(roadmap.size()) % k_neighbors == 0 || iteration == params_.max_iterations);
        if (!query_due || !queryRoadmap()) continue;
        if (first_solution_sample_attempt < 0) {
            first_solution_sample_attempt = result.sample_attempts;
            result.first_solution_time_s = runtime.elapsedSeconds();
            result.first_solution_path_cost = best_cost;
        }
    }

    if (first_solution_sample_attempt >= 0) {
        result.effort_stats.post_solution_sample_attempts = std::max(
            0, result.sample_attempts - first_solution_sample_attempt);
    }

    if (best_path.empty()) queryRoadmap();
    if (best_path.empty()) {
        result.num_nodes = static_cast<int>(roadmap.size());
        result.diagnostics = "prm_queries=" + std::to_string(query_count);
        return finish(false, runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached,
            runtime.deadlineReached() ? "deadline_no_solution" : "PRM failed: no roadmap path found.");
    }

    result.path = std::move(best_path);
    result.path_cost = best_cost;
    result.num_nodes = static_cast<int>(roadmap.size()) + 1 + static_cast<int>(goal_candidates.size());
        result.diagnostics = "prm_queries=" + std::to_string(query_count) +
        " roadmap_nodes=" + std::to_string(roadmap.size());
    return finish(true, PlanningFailureCode::kNone, "");
}

}  // namespace fairino_planning
