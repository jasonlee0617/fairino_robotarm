// myrobot_planning_ros/src/fairino_planner_manager.cpp
// MoveIt2 规划器管理器实现：为 Fairino 机器人提供自定义运动规划算法。
// 支持根据规划组名称自动选择工具模型（法兰/夹爪）

#include "myrobot_planning_ros/fairino_planner_manager.h"
#include <algorithm>
#include <cctype>
#include <pluginlib/class_list_macros.hpp>
#include <moveit/robot_state/conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include "myrobot_planning_core/trajectory/path_shortcut.h"
#include "myrobot_planning_core/trajectory/trajectory_smoother.h"
#include "myrobot_planning_core/constraints/orientation_checker.h"
#include "myrobot_planning_ros/config/parameter_loader.hpp"
#include "myrobot_planning_ros/pipeline/fairino_planning_pipeline.h"

namespace fairino_planning {

namespace {
bool loadFlangeToTool(const moveit::core::RobotModel& model,
                      Transform4d& flange_to_tool) {
    const auto* tool_link = model.getLinkModel("tool0");
    if (!tool_link || !tool_link->getParentLinkModel() ||
        tool_link->getParentLinkModel()->getName() != "wrist3_link") {
        return false;
    }
    Transform4d wrist3_to_tool = Transform4d::Identity();
    wrist3_to_tool.block<3, 3>(0, 0) = tool_link->getJointOriginTransform().linear();
    wrist3_to_tool.block<3, 1>(0, 3) = tool_link->getJointOriginTransform().translation();
    flange_to_tool = DHKinematics::flangeToToolTransform(DHParams{}, wrist3_to_tool);
    return true;
}

std::string normalizePlannerId(const std::string& planner_id) {
    if (planner_id.empty()) {
        return "birrt*";
    }
    std::string key = planner_id;
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::replace(key.begin(), key.end(), '_', '-');

    if (key == "aapf" || key == "aapf-birrt" || key == "aapf-birrt*") {
        return "aapf_birrt*";
    }
    if (key == "mire" || key == "mire-biait" || key == "mire-biait*") {
        return "mire_biait*";
    }
    if (key == "birrt" || key == "birrt*") {
        return "birrt*";
    }
    if (key == "rrt") return "rrt";
    if (key == "rrt*") return "rrt*";
    if (key == "informed-rrt" || key == "informed-rrt*") return "informed_rrt*";
    if (key == "prm") return "prm";
    return planner_id;
}
}  // namespace

// ═══════════════════════════════════════
//  FairinoPlanningContext 实现
// ═══════════════════════════════════════

/// @brief 规划上下文构造函数
/// @param name 上下文名称
/// @param group 规划组名称
/// @param algorithm 实际执行规划的核心算法。
FairinoPlanningContext::FairinoPlanningContext(
    const std::string& name, const std::string& group,
    std::shared_ptr<PlanningAlgorithm> algorithm,
    v2::PipelineOptions pipeline_options)
    : PlanningContext(name, group),
      algorithm_(std::move(algorithm)),
      pipeline_options_(std::move(pipeline_options)),
      cancel_requested_(std::make_shared<std::atomic_bool>(false)) {}

/// @brief 执行规划，填充 MotionPlanResponse
bool FairinoPlanningContext::solve(planning_interface::MotionPlanResponse& res) {
    cancel_requested_->store(false);
    auto options = pipeline_options_;
    options.cancel_requested = [flag = cancel_requested_]() { return flag->load(); };
    v2::FairinoPlanningPipeline pipeline(rclcpp::get_logger("fairino_planner"));
    return pipeline.solve(
        getPlanningScene(),
        getMotionPlanRequest(),
        getGroupName(),
        algorithm_,
        options,
        res);
}

/// @brief 详细响应版本（为兼容 MoveIt 接口，内部调用普通 solve）
bool FairinoPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res) {
    planning_interface::MotionPlanResponse simple_res;
    bool ok = solve(simple_res);
    if (ok) {
        res.trajectory_.push_back(simple_res.trajectory_);
        res.processing_time_.push_back(simple_res.planning_time_);
        res.description_.push_back("FairinoPlanner");
    }
    res.error_code_ = simple_res.error_code_;
    return ok;
}

bool FairinoPlanningContext::terminate() {
    cancel_requested_->store(true);
    return true;
}
void FairinoPlanningContext::clear() { cancel_requested_->store(false); }

// ═══════════════════════════════════════
//  FairinoPlannerManager 实现
// ═══════════════════════════════════════

/// @brief 插件初始化：读取 ROS 参数，配置规划参数
bool FairinoPlannerManager::initialize(
    const moveit::core::RobotModelConstPtr& model,
    const rclcpp::Node::SharedPtr& node,
    const std::string& ns) {

    robot_model_ = model;
    node_ = node;
    if (!robot_model_ || !loadFlangeToTool(*robot_model_, flange_to_tool_)) {
        RCLCPP_ERROR(node_->get_logger(),
            "Fairino planner requires a fixed wrist3_link -> tool0 TCP chain in robot_description.");
        return false;
    }

    planner_config_ = config::loadPlannerConfig(node_, ns);
    aapf_birrt_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.aapf_birrt_star");
    mire_biait_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.mire_biait_star");
    birrt_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.birrt_star");
    rrt_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.rrt");
    rrt_star_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.rrt_star");
    prm_planner_config_ = config::loadPlannerConfig(
        node_, ns, "fairino.algorithms.prm");
    pipeline_options_ = config::loadPipelineOptions(node_, ns);
    pipeline_options_.planner_config = birrt_planner_config_;
    params_ = birrt_planner_config_.planning;

    RCLCPP_INFO(
        node_->get_logger(),
        "Fairino planner params loaded: seed=%u mire_biait*_max_iter=%d aapf_birrt*_max_iter=%d birrt*_max_iter=%d rrt_max_iter=%d rrt*_max_iter=%d prm_max_iter=%d opt=%s",
        pipeline_options_.planner_random_seed,
        mire_biait_planner_config_.planning.max_iterations,
        aapf_birrt_planner_config_.planning.max_iterations,
        birrt_planner_config_.planning.max_iterations,
        rrt_planner_config_.planning.max_iterations,
        rrt_star_planner_config_.planning.max_iterations,
        prm_planner_config_.planning.max_iterations,
        pipeline_options_.enable_path_optimizer ? "on" : "off");

    return true;
}

/// @brief 判断是否能处理给定的规划请求（根据 planner_id）
bool FairinoPlannerManager::canServiceRequest(
    const moveit_msgs::msg::MotionPlanRequest& req) const {
    const auto planner_id = normalizePlannerId(req.planner_id);
    return planner_id == "mire_biait*" ||
           planner_id == "aapf_birrt*" ||
           planner_id == "birrt*" || planner_id == "rrt" || planner_id == "rrt*" ||
           planner_id == "informed_rrt*" || planner_id == "prm";
}

/// @brief 创建规划上下文（核心工厂方法）
planning_interface::PlanningContextPtr FairinoPlannerManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const planning_interface::MotionPlanRequest& req,
    moveit_msgs::msg::MoveItErrorCodes& error_code) const
{
    // 根据请求中的 planner_id 选择算法
    std::shared_ptr<PlanningAlgorithm> algo;
    const auto requested_planner_id = normalizePlannerId(req.planner_id);
    PlannerConfig selected_config;

    if (requested_planner_id == "mire_biait*") {
        algo = std::make_shared<MireBiAitStar>();
        selected_config = mire_biait_planner_config_;
    } else if (requested_planner_id == "aapf_birrt*") {
        algo = std::make_shared<AapfBiRRTStar>();
        selected_config = aapf_birrt_planner_config_;
    } else if (requested_planner_id == "rrt") {
        algo = std::make_shared<RRT>();
        selected_config = rrt_planner_config_;
    } else if (requested_planner_id == "rrt*") {
        algo = std::make_shared<RRTStar>();
        selected_config = rrt_star_planner_config_;
    } else if (requested_planner_id == "informed_rrt*") {
        algo = std::make_shared<InformedRRTStar>();
        selected_config = rrt_star_planner_config_;
    } else if (requested_planner_id == "prm") {
        algo = std::make_shared<PRM>();
        selected_config = prm_planner_config_;
    } else if (requested_planner_id == "birrt*") {
        algo = std::make_shared<BiRRTStar>();
        selected_config = birrt_planner_config_;
    } else {
        RCLCPP_ERROR(
            node_->get_logger(),
            "Unsupported Fairino planner_id='%s'. Use mire_biait*, aapf_birrt*, birrt*, rrt, rrt*, informed_rrt*, or prm.",
            req.planner_id.c_str());
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_MOTION_PLAN;
        return nullptr;
    }

    // 设置规划参数
    algo->configure(selected_config);
    algo->setIKSelectParams(pipeline_options_.ik_selector_params);
    algo->setToolTransform(flange_to_tool_);
    const ToolModel tool_model = ToolModel::GRIPPER;
    algo->setToolModel(tool_model);   // 将工具模型传递给算法

    auto selected_pipeline_options = pipeline_options_;
    selected_pipeline_options.planner_config = selected_config;

    // 创建上下文
    auto context = std::make_shared<FairinoPlanningContext>(
        "fairino_context", req.group_name, algo, selected_pipeline_options);

    // 注入规划场景和请求
    context->setPlanningScene(planning_scene);
    context->setMotionPlanRequest(req);

    // 调试日志：打印使用的工具模型
    RCLCPP_INFO(node_->get_logger(),
                "group_name=%s, selected_planner=%s, requested_planner=%s, tool_model=%s",
                req.group_name.c_str(),
                requested_planner_id.c_str(),
                req.planner_id.c_str(),
                tool_model == ToolModel::GRIPPER ? "GRIPPER" : "FLANGE");

    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return context;
}

}  // namespace fairino_planning

// 注册插件（pluginlib 宏）
PLUGINLIB_EXPORT_CLASS(fairino_planning::FairinoPlannerManager, planning_interface::PlannerManager)
