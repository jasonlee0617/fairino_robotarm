// myrobot_planning_ros/include/myrobot_planning_ros/fairino_planner_manager.h
// MoveIt2 规划器插件管理器：为 Fairino 机器人提供自定义关节空间规划器。

#pragma once

#include <atomic>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <myrobot_planning_core/algorithms/aapf_bi_rrt_star.h>
#include <myrobot_planning_core/algorithms/bi_rrt_star.h>
#include <myrobot_planning_core/algorithms/mire_bi_ait_star.h>
#include <myrobot_planning_core/algorithms/informed_rrt_star.h>
#include <myrobot_planning_core/algorithms/prm.h>
#include <myrobot_planning_core/algorithms/rrt.h>
#include <myrobot_planning_core/algorithms/rrt_star.h>
#include <myrobot_planning_core/ik/fairino_ik.h>
#include <myrobot_planning_core/ik/ik_selector.h>
#include <myrobot_planning_core/dh_kinematics.h>
#include <myrobot_planning_core/types.h>

#include "myrobot_planning_ros/moveit_collision_checker.h"
#include "myrobot_planning_ros/pipeline/fairino_planning_pipeline.h"

namespace fairino_planning {

/// @brief 规划上下文：封装一次具体的规划任务，由 PlannerManager 创建
class FairinoPlanningContext : public planning_interface::PlanningContext {
public:
    /// @param name  上下文名称（通常为规划器名称）
    /// @param group 规划组名称（如 "arm_group"）
    /// @param algorithm 实际执行规划的核心算法。
    FairinoPlanningContext(const std::string& name,
                           const std::string& group,
                           std::shared_ptr<PlanningAlgorithm> algorithm,
                           v2::PipelineOptions pipeline_options);

    /// @brief 执行规划，填充 MotionPlanResponse（包含轨迹、规划时间等）
    bool solve(planning_interface::MotionPlanResponse& res) override;

    /// @brief 执行规划，填充详细响应（包含多种备选方案等）
    bool solve(planning_interface::MotionPlanDetailedResponse& res) override;

    /// @brief 终止正在进行的规划（可用于异步规划）
    bool terminate() override;

    /// @brief 清空上下文状态，准备下一次规划
    void clear() override;

private:
    std::shared_ptr<PlanningAlgorithm> algorithm_;  ///< 持有的规划算法实例
    v2::PipelineOptions pipeline_options_;
    std::shared_ptr<std::atomic_bool> cancel_requested_;
};

/// @brief 规划器管理器：MoveIt2 插件的主入口，负责创建规划上下文
class FairinoPlannerManager : public planning_interface::PlannerManager {
public:
    FairinoPlannerManager() = default;
    ~FairinoPlannerManager() override = default;

    /// @brief 插件初始化，由 MoveIt 在加载时调用
    /// @param model  机器人模型（URDF/SRDF）
    /// @param node   ROS2 节点指针
    /// @param parameter_namespace 参数命名空间（用于读取配置）
    /// @return 初始化是否成功
    bool initialize(const moveit::core::RobotModelConstPtr& model,
                    const rclcpp::Node::SharedPtr& node,
                    const std::string& parameter_namespace) override;

    /// @brief 判断是否能处理给定的规划请求（如检查关节组、约束等）
    bool canServiceRequest(
        const moveit_msgs::msg::MotionPlanRequest& req) const override;

    /// @brief 返回规划器的描述字符串（用于 MoveIt 界面显示）
    std::string getDescription() const override {
        return "Fairino custom joint-space planner";
    }

    /// @brief 返回此规划器支持的所有算法名称（用于 MoveIt 选择）
    void getPlanningAlgorithms(std::vector<std::string>& algs) const override {
        algs.clear();
        algs.push_back("mire_biait*");
        algs.push_back("aapf_birrt*");
        algs.push_back("birrt*");
        algs.push_back("rrt");
        algs.push_back("rrt*");
        algs.push_back("informed_rrt*");
        algs.push_back("prm");
    }

    /// @brief 设置规划器配置（从 ROS 参数服务器读取）
    void setPlannerConfigurations(
        const planning_interface::PlannerConfigurationMap& pcs) override {
        planner_configs_ = pcs;
    }

    /// @brief 获取当前规划器配置（★ 注意：基类中此函数不是虚函数，因此不能加 override）
    const planning_interface::PlannerConfigurationMap&
    getPlannerConfigurations() const {
        return planner_configs_;
    }

    /// @brief 创建规划上下文：根据请求的算法名称，实例化对应的 PlanningAlgorithm
    /// @param planning_scene 当前规划场景（包含障碍物、机器人状态等）
    /// @param req            运动规划请求（起点、终点、约束等）
    /// @param error_code     输出错误码
    /// @return 规划上下文指针（MoveIt 会调用其 solve 方法）
    planning_interface::PlanningContextPtr getPlanningContext(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const planning_interface::MotionPlanRequest& req,
        moveit_msgs::msg::MoveItErrorCodes& error_code) const override;

private:
    rclcpp::Node::SharedPtr node_;                     ///< ROS2 节点
    moveit::core::RobotModelConstPtr robot_model_;     ///< 机器人模型
    PlanningParams params_;                            ///< 默认核心规划参数（步长、迭代次数等）
    PlannerConfig planner_config_;
    PlannerConfig aapf_birrt_planner_config_;
    PlannerConfig mire_biait_planner_config_;
    PlannerConfig birrt_planner_config_;
    PlannerConfig rrt_planner_config_;
    PlannerConfig rrt_star_planner_config_;
    PlannerConfig prm_planner_config_;
    v2::PipelineOptions pipeline_options_;
    Transform4d flange_to_tool_{Transform4d::Identity()};
    planning_interface::PlannerConfigurationMap planner_configs_;  ///< 规划器配置映射
};

}  // namespace fairino_planning
