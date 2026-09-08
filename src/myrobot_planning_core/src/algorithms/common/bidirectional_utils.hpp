#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/config/planning_params.hpp"

namespace fairino_planning::bidirectional {

inline bool isFinite(const JointConfig& config) {
    for (int joint = 0; joint < NUM_JOINTS; ++joint) {
        if (!std::isfinite(config[joint])) return false;
    }
    return true;
}

inline std::vector<ObstacleInfo> normalizeObstacles(
    const Vector3d& origin, const Vector3d& size, const std::vector<ObstacleInfo>& obstacles) {
    std::vector<ObstacleInfo> result;
    for (const auto& obstacle : obstacles) {
        if (obstacle.size.cwiseAbs().maxCoeff() > 1e-9) result.push_back(obstacle);
    }
    if (result.empty() && size.cwiseAbs().maxCoeff() > 1e-9) {
        result.push_back(ObstacleInfo{origin, size});
    }
    return result;
}

inline double rewireRadius(const PlanningParams& params, int nodes, double minimum_step_scale = 1.2) {
    const double count = static_cast<double>(std::max(nodes, 2));
    const double radius = params.gamma * std::pow(std::log(count) / count, 1.0 / NUM_JOINTS);
    return std::min(params.max_rewire_radius, std::max(radius, params.max_step * minimum_step_scale));
}

inline bool strictlyValid(
    const CollisionInterface& collision, const std::vector<JointConfig>& path,
    double validation_distance, int* bad_segment = nullptr) {
    if (bad_segment) *bad_segment = -1;
    if (path.empty() || !isFinite(path.front()) || !collision.isStateValid(path.front())) {
        if (bad_segment) *bad_segment = 0;
        return false;
    }
    for (size_t index = 1; index < path.size(); ++index) {
        if (!isFinite(path[index]) || !collision.isStateValid(path[index]) ||
            !collision.isMotionValid(path[index - 1U], path[index], validation_distance)) {
            if (bad_segment) *bad_segment = static_cast<int>(index - 1U);
            return false;
        }
    }
    return true;
}

}  // namespace fairino_planning::bidirectional
