// include/myrobot_planning_ros/moveit_collision_checker.h
#pragma once
#include "myrobot_planning_core/collision/collision_interface.h"
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <string>
#include <vector>

namespace fairino_planning {

class MoveItCollisionChecker : public CollisionInterface {

public:
    //接收一个只读的 PlanningScene 智能指针（ConstPtr）和规划组名称。PlanningScene 是 MoveIt2 的核心类，包含机器人模型、环境障碍物、碰撞世界等信息
    MoveItCollisionChecker(
        const planning_scene::PlanningSceneConstPtr& scene,//保存传入的规划场景指针，用于碰撞检测
        const std::string& group_name);//规划组名
    //使用 override 关键字明确表示重写基类虚函数
    bool isStateValid(const JointConfig& q) const override;
    bool isMotionValid(const JointConfig& q1, const JointConfig& q2,
                       double validation_distance = 0.10) const override;
    bool supportsWorldClearance() const override { return true; }
    ClearanceQueryResult nearestWorldClearance(
        const JointConfig& q, double influence_distance) const override;
    std::vector<bool> areStatesValid(
        const std::vector<JointConfig>& states) const override;

private:
    planning_scene::PlanningSceneConstPtr scene_;
    std::string group_name_;
    const moveit::core::JointModelGroup* jmg_;

    bool setJointValues(moveit::core::RobotState& state, const JointConfig& q) const;
};

}  // namespace fairino_planning
