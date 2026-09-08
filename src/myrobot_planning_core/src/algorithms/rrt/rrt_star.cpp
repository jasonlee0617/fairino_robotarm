// src/algorithms/rrt/rrt_star.cpp
#include "myrobot_planning_core/algorithms/rrt_star.h"
#include "rrt_star_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fairino_planning {
namespace {

bool meaningfulObstacle(const ObstacleInfo& obs) {
    return obs.size.cwiseAbs().maxCoeff() > 1e-9;
}

std::vector<ObstacleInfo> normalizeObstacles(
    const Vector3d& obs_origin, const Vector3d& obs_size,
    const std::vector<ObstacleInfo>& obstacles) {
    std::vector<ObstacleInfo> out;
    for (const auto& obs : obstacles) {
        if (meaningfulObstacle(obs)) out.push_back(obs);
    }
    if (out.empty()) {
        ObstacleInfo single{obs_origin, obs_size};
        if (meaningfulObstacle(single)) out.push_back(single);
    }
    return out;
}

}  // namespace

RRTStar::RRTStar() : rng_(42) {}

PlanResult RRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    const auto obstacles = normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles);
    return planOnce(request.q_start,
                    request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
                    request.p_start, request.p_goal,
                    request.R_target, obstacles, request.random_seed, &request);
}

PlanResult RRTStar::plan(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const Vector3d& obs_origin, const Vector3d& obs_size) {
    const auto obstacles = normalizeObstacles(obs_origin, obs_size, {});
    return planOnce(q_start, {q_goal}, p_start, p_goal, R_target, obstacles, 0);
}

PlanResult RRTStar::planMultiObs(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const std::vector<ObstacleInfo>& obstacles) {
    return planOnce(q_start, {q_goal}, p_start, p_goal, R_target, obstacles, 0);
}

PlanResult RRTStar::planOnce(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles,
    unsigned int request_seed,
    const PlanRequestCore* request) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obstacles;

    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0 ? 42U : request_seed));
    if (!collision_) {
        PlanResult result;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "RRT*: null collision checker.";
        return result;
    }
    return runRrtStarSearch(
        q_start, goal_candidates, params_, limits_, *collision_,
        [this, &goal_candidates](const RrtStarSearchState&) {
            JointConfig sample = limits_.sampleUniform(rng_);
            if (params_.goal_bias > 0.0) {
                std::uniform_real_distribution<double> coin(0.0, 1.0);
                if (coin(rng_) < params_.goal_bias) {
                    std::uniform_int_distribution<size_t> pick(0U, goal_candidates.size() - 1U);
                    sample = goal_candidates[pick(rng_)];
                }
            }
            return sample;
        },
        "RRT*", request);
}

}  // namespace fairino_planning
