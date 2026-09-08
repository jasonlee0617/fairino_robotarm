#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/types/aliases.hpp"

namespace fairino_planning::goal_set {

constexpr double kEps = 1e-10;

inline double distance(const JointConfig& lhs, const JointConfig& rhs) {
    return (lhs - rhs).norm();
}

inline double lowerBound(const JointConfig& start, const std::vector<JointConfig>& goals) {
    double bound = std::numeric_limits<double>::infinity();
    for (const auto& goal : goals) bound = std::min(bound, distance(start, goal));
    return bound;
}

inline bool strictlyValid(
    const CollisionInterface& collision, const std::vector<JointConfig>& path,
    double validation_distance) {
    if (path.empty() || !collision.isStateValid(path.front())) return false;
    for (size_t index = 1; index < path.size(); ++index) {
        if (!collision.isStateValid(path[index]) ||
            !collision.isMotionValid(path[index - 1U], path[index], validation_distance)) {
            return false;
        }
    }
    return true;
}

struct DirectPath {
    std::vector<JointConfig> path;
    double cost = std::numeric_limits<double>::infinity();
    int goal_index = -1;

    bool reachesLowerBound(double lower_bound) const {
        return std::isfinite(cost) && std::abs(cost - lower_bound) <= kEps;
    }
};

inline std::optional<DirectPath> shortestStrictDirectPath(
    const CollisionInterface& collision, const JointConfig& start,
    const std::vector<JointConfig>& goals, double validation_distance) {
    std::optional<DirectPath> best;
    for (size_t index = 0; index < goals.size(); ++index) {
        const double cost = distance(start, goals[index]);
        if (best && cost + kEps >= best->cost) continue;
        std::vector<JointConfig> path{start, goals[index]};
        if (!strictlyValid(collision, path, validation_distance)) continue;
        best = DirectPath{std::move(path), cost, static_cast<int>(index)};
    }
    return best;
}

}  // namespace fairino_planning::goal_set
