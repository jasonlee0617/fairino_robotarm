#pragma once

#include <functional>
#include <random>
#include <string>
#include <vector>

#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/config/planning_params.hpp"
#include "myrobot_planning_core/result/plan_result.hpp"

namespace fairino_planning {

struct RrtStarSearchState {
    bool has_solution = false;
    double best_cost = 0.0;
};

using RrtStarSampler = std::function<JointConfig(const RrtStarSearchState&)>;

PlanResult runRrtStarSearch(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    RrtStarSampler sampler,
    const std::string& planner_name,
    const PlanRequestCore* request = nullptr);

}  // namespace fairino_planning
