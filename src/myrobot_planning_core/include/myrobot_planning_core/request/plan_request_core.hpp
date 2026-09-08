#pragma once

#include <chrono>
#include <functional>
#include <vector>

#include "myrobot_planning_core/config/planning_params.hpp"
#include "myrobot_planning_core/model/robot_kinematics_config.hpp"

namespace fairino_planning {

struct PlanRequestCore {
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    // Ordered, collision-free terminal roots for a pose goal.  Empty preserves
    // the legacy single-goal contract and means {q_goal}.
    std::vector<JointConfig> goal_candidates;
    Vector3d p_start = Vector3d::Zero();
    Vector3d p_goal = Vector3d::Zero();
    RotMatrix3d R_target = RotMatrix3d::Identity();
    // Legacy single-obstacle fields for backward compatibility.
    Vector3d obs_origin = Vector3d::Zero();
    Vector3d obs_size = Vector3d::Zero();
    // Preferred obstacle input path: full obstacle list + multi-obstacle switch.
    std::vector<ObstacleInfo> obstacles;
    ToolModel tool_model = ToolModel::FLANGE;
    unsigned int random_seed = 0;
    bool use_multi_obstacle = false;  // if true (or obstacles non-empty), planners should use multi-obstacle mode first.

    // Absolute benchmark deadline: IK/root enumeration and the core search
    // share one wall-clock budget. The default keeps ordinary calls bounded
    // only by max_iterations.
    std::chrono::steady_clock::time_point started{};
    std::chrono::steady_clock::time_point deadline{};
    std::function<bool()> cancel_requested;
    std::vector<double> anytime_checkpoints_s{0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0};

    bool hasDeadline() const {
        return deadline != std::chrono::steady_clock::time_point{};
    }

    bool shouldStop() const {
        return (cancel_requested && cancel_requested()) ||
            (hasDeadline() && std::chrono::steady_clock::now() >= deadline);
    }
};

}  // namespace fairino_planning
