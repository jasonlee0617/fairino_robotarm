// src/moveit_collision_checker.cpp
#include "myrobot_planning_ros/moveit_collision_checker.h"
#include <cmath>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_env.h>

namespace fairino_planning {

MoveItCollisionChecker::MoveItCollisionChecker(
    const planning_scene::PlanningSceneConstPtr& scene,//保存场景指针
    const std::string& group_name)//保存规划组名
    : scene_(scene),
      group_name_(group_name),
      jmg_(nullptr) {//从机器人模型中获取指定规划组的 JointModelGroup 指针，用于后续操作
    if (scene_ && scene_->getRobotModel()) {
        jmg_ = scene_->getRobotModel()->getJointModelGroup(group_name_);
    }
}

bool MoveItCollisionChecker::setJointValues(
    moveit::core::RobotState& state, const JointConfig& q) const {
    if (!jmg_ || jmg_->getVariableCount() != NUM_JOINTS) {
        return false;
    }

    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) {
            return false;
        }
    }

    std::vector<double> values(q.data(), q.data() + NUM_JOINTS);//返回 Eigen 向量内部数组的指针，q.data() + NUM_JOINTS 指向末尾，因此构造的 std::vector<double> 包含了 q 的所有元素
    state.setJointGroupPositions(jmg_, values);//将 values 设置到指定关节组的所有关节上
    state.update();//根据关节值重新计算所有连杆的位姿变换，保证后续碰撞检测使用最新的运动学信息
    return state.satisfiesBounds(jmg_);
}

bool MoveItCollisionChecker::isStateValid(const JointConfig& q) const {
    if (!scene_ || !scene_->getRobotModel() || !jmg_) {
        return false;
    }

    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) {
            return false;
        }
    }

    moveit::core::RobotState robot_state(scene_->getCurrentState());
    if (!setJointValues(robot_state, q)) {//设置状态：setJointValues(robot_state_, q) 更新 robot_state_ 为当前关节角
        return false;
    }

    // 碰撞检查
    collision_detection::CollisionRequest req;//创建 CollisionRequest 对象，设置 group_name 为当前规划组
    collision_detection::CollisionResult res;//CollisionResult 用于存储结果
    req.group_name = group_name_;
    scene_->checkCollision(req, res, robot_state);//调用 scene_->checkCollision(req, res, robot_state_)，由 MoveIt2 执行实际的碰撞检测
    return !res.collision;
}

bool MoveItCollisionChecker::isMotionValid(
    const JointConfig& q1, const JointConfig& q2,
    double validation_distance) const {

    if (!scene_ || !jmg_ || validation_distance <= 0.0 ||
        !std::isfinite(validation_distance)) {
        return false;
    }

    // Match MoveIt's exported trajectory: bounded joints are interpolated
    // between their requested values, not along a wrapped shortcut.
    const JointConfig dq = q2 - q1;
    const double d = dq.norm();
    int n_steps = std::max(2, static_cast<int>(std::ceil(d / validation_distance)) + 1);

    for (int s = 1; s <= n_steps; ++s) {
        double alpha = static_cast<double>(s) / n_steps;
        JointConfig q_interp = q1 + alpha * dq;
        if (!isStateValid(q_interp))
            return false;
    }
    return true;
}

ClearanceQueryResult MoveItCollisionChecker::nearestWorldClearance(
    const JointConfig& q, double influence_distance) const {
    ClearanceQueryResult out;
    out.supported = true;
    if (!scene_ || !scene_->getRobotModel() || !jmg_ ||
        !std::isfinite(influence_distance) || influence_distance <= 0.0) {
        out.supported = false;
        return out;
    }

    moveit::core::RobotState robot_state(scene_->getCurrentState());
    if (!setJointValues(robot_state, q)) {
        return out;
    }

    collision_detection::DistanceRequest request;
    request.type = collision_detection::DistanceRequestTypes::SINGLE;
    request.group_name = group_name_;
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.distance_threshold = influence_distance;
    request.max_contacts_per_body = 1;
    request.acm = &scene_->getAllowedCollisionMatrix();
    request.enableGroup(scene_->getRobotModel());

    collision_detection::DistanceResult result;
    scene_->getCollisionEnv()->distanceRobot(request, result, robot_state);
    const auto& nearest = result.minimum_distance;
    if (!std::isfinite(nearest.distance) || nearest.distance > influence_distance) {
        return out;
    }

    int robot_index = -1;
    if (nearest.body_types[0] == collision_detection::BodyType::ROBOT_LINK) {
        robot_index = 0;
    } else if (nearest.body_types[1] == collision_detection::BodyType::ROBOT_LINK) {
        robot_index = 1;
    }
    if (robot_index < 0 || nearest.link_names[robot_index].empty()) {
        return out;
    }

    const auto* link = scene_->getRobotModel()->getLinkModel(
        nearest.link_names[robot_index]);
    if (!link) {
        return out;
    }

    const int obstacle_index = 1 - robot_index;
    const Vector3d robot_point = nearest.nearest_points[robot_index];
    const Vector3d obstacle_point = nearest.nearest_points[obstacle_index];
    Vector3d away = robot_point - obstacle_point;
    if (away.norm() <= 1e-9) {
        away = robot_index == 0 ? -nearest.normal : nearest.normal;
    }
    if (!away.allFinite() || away.norm() <= 1e-9) {
        return out;
    }
    away.normalize();

    const Eigen::Isometry3d& link_transform = robot_state.getGlobalLinkTransform(link);
    const Vector3d point_in_link = link_transform.inverse() * robot_point;
    Eigen::MatrixXd jacobian;
    if (!robot_state.getJacobian(jmg_, link, point_in_link, jacobian, false) ||
        jacobian.rows() < 3 || jacobian.cols() != NUM_JOINTS) {
        return out;
    }

    const Eigen::VectorXd gradient = jacobian.topRows(3).transpose() * away;
    if (!gradient.allFinite()) {
        return out;
    }
    out.has_obstacle = true;
    out.signed_distance = nearest.distance;
    out.joint_gradient = gradient;
    out.robot_link = nearest.link_names[robot_index];
    return out;
}

std::vector<bool> MoveItCollisionChecker::areStatesValid(
    const std::vector<JointConfig>& states) const {
    std::vector<bool> out;
    out.reserve(states.size());
    for (const auto& s : states) {
        out.push_back(isStateValid(s));
    }
    return out;
}

}  // namespace fairino_planning
