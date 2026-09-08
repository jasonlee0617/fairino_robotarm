// include/myrobot_planning_core/collision/collision_interface.h
#pragma once
#include "myrobot_planning_core/types.h"
#include <limits>
#include <string>
#include <vector>

namespace fairino_planning {

struct ClearanceQueryResult {
    bool supported = false;
    bool has_obstacle = false;
    double signed_distance = std::numeric_limits<double>::infinity();
    JointConfig joint_gradient = JointConfig::Zero();
    std::string robot_link;
};

// 纯抽象接口 — 核心库不依赖 ROS
class CollisionInterface {
public:
    virtual ~CollisionInterface() = default;

    // 纯虚函数，检查给定关节配置是否有效（不碰撞、不超出限位等）。const 表示该操作不修改对象状态
    virtual bool isStateValid(const JointConfig& q) const = 0;

    // 检查从 q1 到 q2 的运动段是否有效。通常会在线段上插值多个点进行检测，validation_distance 指定插值步长，确保路径的连续安全性
    virtual bool isMotionValid(const JointConfig& q1, const JointConfig& q2,
                               double validation_distance = 0.10) const = 0;

    // Optional capability used by link-aware planners. The returned gradient
    // points in the joint-space direction that increases robot-world clearance.
    virtual bool supportsWorldClearance() const { return false; }
    virtual ClearanceQueryResult nearestWorldClearance(
        const JointConfig&, double) const {
        return {};
    }

    // 可选：批量状态检查（默认逐个调用，便于后端优化）
    virtual std::vector<bool> areStatesValid(
        const std::vector<JointConfig>& states) const {
        std::vector<bool> out;
        out.reserve(states.size());
        for (const auto& s : states) {
            out.push_back(isStateValid(s));
        }
        return out;
    }
};

}  // namespace fairino_planning
