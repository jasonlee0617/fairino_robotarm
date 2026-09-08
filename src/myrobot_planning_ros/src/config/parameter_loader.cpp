#include "myrobot_planning_ros/config/parameter_loader.hpp"
#include "myrobot_planning_core/config/planning_params.hpp"
#include "myrobot_planning_ros/pipeline/fairino_planning_pipeline.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace fairino_planning::config {
namespace {

std::string scoped(const std::string& ns, const std::string& key) {
    if (ns.empty() || key.rfind(ns + ".", 0) == 0) return key;
    return ns + "." + key;
}
double gd(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, double d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<double>(name, d);
    return n->get_parameter(name).as_double();
}
bool gb(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, bool d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<bool>(name, d);
    return n->get_parameter(name).as_bool();
}
int gi(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, int d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<int>(name, d);
    return n->get_parameter(name).as_int();
}
std::string gs(const rclcpp::Node::SharedPtr& n, const std::string& ns, const std::string& k, const std::string& d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<std::string>(name, d);
    return n->get_parameter(name).as_string();
}
std::vector<double> gda(const rclcpp::Node::SharedPtr& n, const std::string& ns,
                         const std::string& k, const std::vector<double>& d) {
    auto name = scoped(ns, k); if (!n->has_parameter(name)) n->declare_parameter<std::vector<double>>(name, d);
    return n->get_parameter(name).as_double_array();
}

std::string prefixed(const std::string& prefix, const std::string& key) {
    return prefix.empty() ? key : prefix + "." + key;
}

double gd_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, double d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gd(n, ns, primary, d);
}

bool gb_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, bool d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gb(n, ns, primary, d);
}

int gi_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, int d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gi(n, ns, primary, d);
}

std::vector<double> gda_pref(
    const rclcpp::Node::SharedPtr& n, const std::string& ns,
    const std::string& primary_prefix, const std::string& legacy_prefix,
    const std::string& k, const std::vector<double>& d) {
    (void)legacy_prefix;
    const auto primary = prefixed(primary_prefix, k);
    return gda(n, ns, primary, d);
}

Vector3d vector3From(const std::vector<double>& values, const Vector3d& fallback) {
    if (values.size() != 3U) return fallback;
    return Vector3d(values[0], values[1], values[2]);
}

ToolParams loadGripperToolParams(
    const rclcpp::Node::SharedPtr& node,
    const std::string& ns,
    const ToolParams& fallback) {
    ToolParams params = fallback;
    params.offset = vector3From(
        gda(node, ns, "fairino.ik.tool.gripper.xyz",
            {fallback.offset.x(), fallback.offset.y(), fallback.offset.z()}),
        fallback.offset);
    params.rpy = vector3From(
        gda(node, ns, "fairino.ik.tool.gripper.rpy",
            {fallback.rpy.x(), fallback.rpy.y(), fallback.rpy.z()}),
        fallback.rpy);
    return params;
}

// W_move helpers
std::vector<double> wmDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.W_move(i,i); return o; }
void awm(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; p.W_move.setZero(); for(int i=0;i<NUM_JOINTS;++i)p.W_move(i,i)=v[i]; }

// seed_delta helpers
std::vector<double> ssDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_start[i]; return o; }
void ass(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_start[i]=v[i]; }
std::vector<double> swDef(const IKSelectParams& p) { std::vector<double> o(NUM_JOINTS); for(int i=0;i<NUM_JOINTS;++i)o[i]=p.seed_delta_soft_weight[i]; return o; }
void asw(IKSelectParams& p, const std::vector<double>& v) { if(v.size()!=NUM_JOINTS)return; for(int i=0;i<NUM_JOINTS;++i)p.seed_delta_soft_weight[i]=v[i]; }

IKTaskProfile parseTaskProfile(std::string value, IKTaskProfile fallback) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "grasp" || value == "industrial_grasp") return IKTaskProfile::Grasp;
    if (value == "continuous" || value == "cartesian" || value == "servo") {
        return IKTaskProfile::Continuous;
    }
    return fallback;
}

}  // namespace

IKSelectParams loadIKSelectParams(const rclcpp::Node::SharedPtr& node, const std::string& ns)
{
    IKSelectParams p;
    p.gripper_tool = loadGripperToolParams(node, ns, p.gripper_tool);
    p.task_profile = parseTaskProfile(
        gs(node, ns, "fairino.ik.task_profile", toString(p.task_profile)),
        p.task_profile);

    // 1. manipulability
    p.mu_eps               = gd(node, ns, "fairino.ik.manipulability.mu_eps", p.mu_eps);
    p.alpha_manipulability = gd(node, ns, "fairino.ik.manipulability.alpha_manipulability", p.alpha_manipulability);
    p.sigma_hard_flange    = gd(node, ns, "fairino.ik.manipulability.sigma_hard_flange", p.sigma_hard_flange);
    p.sigma_hard_gripper   = gd(node, ns, "fairino.ik.manipulability.sigma_hard_gripper", p.sigma_hard_gripper);
    p.cond_hard_max        = gd(node, ns, "fairino.ik.manipulability.cond_hard_max", p.cond_hard_max);
    p.sigma_min_threshold  = gd(node, ns, "fairino.ik.manipulability.sigma_min_threshold", p.sigma_min_threshold);

    // 2. continuity
    awm(p, gda(node, ns, "fairino.ik.continuity.W_move_diag", wmDef(p)));
    p.alpha_continuity  = gd(node, ns, "fairino.ik.continuity.alpha_continuity", p.alpha_continuity);
    p.cost_eps          = gd(node, ns, "fairino.ik.continuity.cost_eps", p.cost_eps);
    p.lexicographic_eps = gd(node, ns, "fairino.ik.continuity.lexicographic_eps", p.lexicographic_eps);
    p.enable_continuity_guard = gb(node, ns, "fairino.ik.continuity.enable_continuity_guard", p.enable_continuity_guard);
    p.max_joint_step_rad = gd(node, ns, "fairino.ik.continuity.max_joint_step_rad", p.max_joint_step_rad);
    p.max_wrist_step_rad = gd(node, ns, "fairino.ik.continuity.max_wrist_step_rad", p.max_wrist_step_rad);
    p.branch_switch_hard_reject = gb(node, ns, "fairino.ik.continuity.branch_switch_hard_reject", p.branch_switch_hard_reject);
    p.branch_switch_min_step_rad = gd(node, ns, "fairino.ik.continuity.branch_switch_min_step_rad", p.branch_switch_min_step_rad);
    p.hint_seed_sync_max_rad = gd(node, ns, "fairino.ik.continuity.hint_seed_sync_max_rad", p.hint_seed_sync_max_rad);
    p.cartesian_stream_max_pos_step_m = gd(
        node, ns, "fairino.ik.continuity.cartesian_stream_max_pos_step_m", p.cartesian_stream_max_pos_step_m);
    p.cartesian_stream_max_rot_step_rad = gd(
        node, ns, "fairino.ik.continuity.cartesian_stream_max_rot_step_rad", p.cartesian_stream_max_rot_step_rad);
    p.continuous_max_goal_roots = gi(
        node, ns, "fairino.ik.continuous.max_goal_roots", p.continuous_max_goal_roots);
    p.continuous_goal_root_min_separation_rad = gd(
        node, ns, "fairino.ik.continuous.goal_root_min_separation_rad",
        p.continuous_goal_root_min_separation_rad);

    // 3. posture
    p.upper_arm_min_z_soft   = gd(node, ns, "fairino.ik.posture.upper_arm_min_z_soft", p.upper_arm_min_z_soft);
    p.upper_arm_min_z_hard   = gd(node, ns, "fairino.ik.posture.upper_arm_min_z_hard", p.upper_arm_min_z_hard);
    p.forearm_min_z_soft     = gd(node, ns, "fairino.ik.posture.forearm_min_z_soft", p.forearm_min_z_soft);
    p.forearm_min_z_hard     = gd(node, ns, "fairino.ik.posture.forearm_min_z_hard", p.forearm_min_z_hard);
    p.wrist_chain_min_z_soft = gd(node, ns, "fairino.ik.posture.wrist_chain_min_z_soft", p.wrist_chain_min_z_soft);
    p.wrist_chain_min_z_hard = gd(node, ns, "fairino.ik.posture.wrist_chain_min_z_hard", p.wrist_chain_min_z_hard);
    p.anti_gravity_soft      = gd(node, ns, "fairino.ik.posture.anti_gravity_soft", p.anti_gravity_soft);
    p.anti_gravity_hard      = gd(node, ns, "fairino.ik.posture.anti_gravity_hard", p.anti_gravity_hard);

    // 4. wrist
    p.q4_inner_soft_start    = gd(node, ns, "fairino.ik.wrist.q4_inner_soft_start", p.q4_inner_soft_start);
    p.q4_inner_hard_max      = gd(node, ns, "fairino.ik.wrist.q4_inner_hard_max", p.q4_inner_hard_max);
    p.alpha_q4_inner         = gd(node, ns, "fairino.ik.wrist.alpha_q4_inner", p.alpha_q4_inner);
    p.q4_positive_weight     = gd(node, ns, "fairino.ik.wrist.q4_positive_weight", p.q4_positive_weight);
    p.q5_ref                 = gd(node, ns, "fairino.ik.wrist.q5_ref", p.q5_ref);
    p.q5_ref_weight          = gd(node, ns, "fairino.ik.wrist.q5_ref_weight", p.q5_ref_weight);
    p.forearm_tool_angle_soft = gd(node, ns, "fairino.ik.wrist.forearm_tool_angle_soft", p.forearm_tool_angle_soft);
    p.forearm_tool_angle_hard = gd(node, ns, "fairino.ik.wrist.forearm_tool_angle_hard", p.forearm_tool_angle_hard);
    p.alpha_wrist_fold       = gd(node, ns, "fairino.ik.wrist.alpha_wrist_fold", p.alpha_wrist_fold);

    // 5. joint safety
    p.joint_margin_hard_rad = gd(node, ns, "fairino.ik.joint_safety.joint_margin_hard_rad", p.joint_margin_hard_rad);
    p.wrist_sin_min         = gd(node, ns, "fairino.ik.joint_safety.wrist_sin_min", p.wrist_sin_min);
    p.elbow_sin_min         = gd(node, ns, "fairino.ik.joint_safety.elbow_sin_min", p.elbow_sin_min);
    p.base_radius_min       = gd(node, ns, "fairino.ik.joint_safety.base_radius_min", p.base_radius_min);
    p.reject_q2_positive    = gb(node, ns, "fairino.ik.joint_safety.reject_q2_positive", p.reject_q2_positive);
    p.reject_q4_positive    = gb(node, ns, "fairino.ik.joint_safety.reject_q4_positive", p.reject_q4_positive);

    // 6. scoring weights
    p.S1_continuity     = gd(node, ns, "fairino.ik.scoring_weights.S1_continuity", p.S1_continuity);
    p.S2_manipulability = gd(node, ns, "fairino.ik.scoring_weights.S2_manipulability", p.S2_manipulability);
    p.S3_posture        = gd(node, ns, "fairino.ik.scoring_weights.S3_posture", p.S3_posture);
    p.S4_joint_safety   = gd(node, ns, "fairino.ik.scoring_weights.S4_joint_safety", p.S4_joint_safety);

    // 7. seed delta
    p.enable_seed_delta_hard_filter = gb(node, ns, "fairino.ik.seed_delta.enable_hard_filter", p.enable_seed_delta_hard_filter);
    p.allow_large_motion_fallback   = gb(node, ns, "fairino.ik.seed_delta.allow_large_motion_fallback", p.allow_large_motion_fallback);
    ass(p, gda(node, ns, "fairino.ik.seed_delta.soft_start", ssDef(p)));
    asw(p, gda(node, ns, "fairino.ik.seed_delta.soft_weight", swDef(p)));

    // 8. debug
    p.debug_log_all_candidates    = gb(node, ns, "fairino.ik.debug.log_all_candidates", p.debug_log_all_candidates);
    p.debug_max_candidates_to_log = gi(node, ns, "fairino.ik.debug.max_candidates_to_log", p.debug_max_candidates_to_log);
    p.debug_print_degrees         = gb(node, ns, "fairino.ik.debug.print_degrees", p.debug_print_degrees);
    p.debug_log_every_n_calls     = gi(node, ns, "fairino.ik.debug.log_every_n_calls", p.debug_log_every_n_calls);
    p.debug_always_log_failures   = gb(node, ns, "fairino.ik.debug.always_log_failures", p.debug_always_log_failures);

    // 9. task profiles
    p.grasp_hard_reject_low_arm = gb(
        node, ns, "fairino.ik.grasp.hard_reject_low_arm", p.grasp_hard_reject_low_arm);
    p.grasp_hard_reject_wrist_fold = gb(
        node, ns, "fairino.ik.grasp.hard_reject_wrist_fold", p.grasp_hard_reject_wrist_fold);
    p.grasp_allow_industrial_fallback = gb(
        node, ns, "fairino.ik.grasp.allow_industrial_fallback", p.grasp_allow_industrial_fallback);
    p.grasp_upper_arm_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.upper_arm_min_z_hard", p.grasp_upper_arm_min_z_hard);
    p.grasp_forearm_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.forearm_min_z_hard", p.grasp_forearm_min_z_hard);
    p.grasp_wrist_chain_min_z_hard = gd(
        node, ns, "fairino.ik.grasp.wrist_chain_min_z_hard", p.grasp_wrist_chain_min_z_hard);
    p.grasp_q4_inner_hard_max = gd(
        node, ns, "fairino.ik.grasp.q4_inner_hard_max", p.grasp_q4_inner_hard_max);
    p.grasp_forearm_tool_angle_hard = gd(
        node, ns, "fairino.ik.grasp.forearm_tool_angle_hard", p.grasp_forearm_tool_angle_hard);
    p.continuous_enforce_branch_guard = gb(
        node, ns, "fairino.ik.continuous.enforce_branch_guard", p.continuous_enforce_branch_guard);
    p.continuous_enforce_consistency_limits = gb(
        node, ns, "fairino.ik.continuous.enforce_consistency_limits",
        p.continuous_enforce_consistency_limits);

    return p;
}

AnalyticalIKParams loadAnalyticalIKParams(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    AnalyticalIKParams p;
    p.gripper_tool = loadGripperToolParams(node, ns, p.gripper_tool);
    p.rho_sq_neg_eps = gd(node, ns, "fairino.ik.analytical.rho_sq_neg_eps", p.rho_sq_neg_eps);
    p.wrist_singularity_s5_min = gd(node, ns, "fairino.ik.analytical.wrist_singularity_s5_min", p.wrist_singularity_s5_min);
    p.D_domain_eps = gd(node, ns, "fairino.ik.analytical.D_domain_eps", p.D_domain_eps);
    p.fk_verify_pos_tol = gd(node, ns, "fairino.ik.analytical.fk_verify_pos_tol", p.fk_verify_pos_tol);
    p.fk_verify_rot_tol = gd(node, ns, "fairino.ik.analytical.fk_verify_rot_tol", p.fk_verify_rot_tol);
    p.solution_unique_tol = gd(node, ns, "fairino.ik.analytical.solution_unique_tol", p.solution_unique_tol);
    p.candidate_dup_norm_tol = gd(node, ns, "fairino.ik.analytical.candidate_dup_norm_tol", p.candidate_dup_norm_tol);
    p.log_threshold_summary = gb(node, ns, "fairino.ik.analytical.log_threshold_summary", p.log_threshold_summary);
    p.log_stage_survival = gb(node, ns, "fairino.ik.analytical.log_stage_survival", p.log_stage_survival);
    return p;
}

PlannerConfig loadPlannerConfig(
    const rclcpp::Node::SharedPtr& node,
    const std::string& ns,
    const std::string& planner_parameter_namespace) {
    PlannerConfig cfg;
    const std::string prefix = planner_parameter_namespace.empty() ? "fairino" : planner_parameter_namespace;
    constexpr const char* legacy = "fairino";

    auto& p = cfg.planning;
    p.max_iterations = gi_pref(node, ns, prefix, legacy, "max_iterations", p.max_iterations);
    p.post_solution_sample_attempts = gi_pref(
        node, ns, prefix, legacy, "post_solution_sample_attempts",
        p.post_solution_sample_attempts);
    p.max_step = gd_pref(node, ns, prefix, legacy, "max_step", p.max_step);
    p.goal_threshold = gd_pref(node, ns, prefix, legacy, "goal_threshold", p.goal_threshold);
    p.goal_bias = gd_pref(node, ns, prefix, legacy, "goal_bias", p.goal_bias);
    p.max_ik_tries = gi_pref(node, ns, prefix, legacy, "max_ik_tries", p.max_ik_tries);
    p.gamma = gd_pref(node, ns, prefix, legacy, "gamma", p.gamma);
    p.max_rewire_radius = gd_pref(node, ns, prefix, legacy, "max_rewire_radius", p.max_rewire_radius);
    p.max_near = gi_pref(node, ns, prefix, legacy, "max_near", p.max_near);
    p.connect_max_steps = gi_pref(node, ns, prefix, legacy, "connect_max_steps", p.connect_max_steps);
    p.connect_goal_bias = gd_pref(node, ns, prefix, legacy, "connect_goal_bias", p.connect_goal_bias);
    p.rewire_every_k = gi_pref(node, ns, prefix, legacy, "rewire_every_k", p.rewire_every_k);
    p.rewire_max_neighbors = gi_pref(node, ns, prefix, legacy, "rewire_max_neighbors", p.rewire_max_neighbors);
    p.validation_distance = gd_pref(node, ns, prefix, legacy, "validation_distance", p.validation_distance);
    p.mire_biait.batch_size = gi_pref(
        node, ns, prefix, legacy, "batch_size", p.mire_biait.batch_size);
    p.mire_biait.post_solution_batch_size = gi_pref(
        node, ns, prefix, legacy, "post_solution_batch_size",
        p.mire_biait.post_solution_batch_size);
    p.mire_biait.rgg_factor = gd_pref(
        node, ns, prefix, legacy, "rgg_factor", p.mire_biait.rgg_factor);
    p.mire_biait.effort_focal_factor = gd_pref(
        node, ns, prefix, legacy, "effort_focal_factor", p.mire_biait.effort_focal_factor);
    p.mire_biait.ablation_variant = gs(
        node, ns, prefixed(prefix, "ablation_variant"), p.mire_biait.ablation_variant);
    p.mire_biait.enable_effort_focal_queue = gb_pref(
        node, ns, prefix, legacy, "enable_effort_focal_queue", p.mire_biait.enable_effort_focal_queue);
    p.mire_biait.enable_lazy_edge_validation = gb_pref(
        node, ns, prefix, legacy, "enable_lazy_edge_validation", p.mire_biait.enable_lazy_edge_validation);
    p.prm.k_neighbors = gi_pref(node, ns, prefix, legacy, "k_neighbors", p.prm.k_neighbors);

    p.connect_success_every_k = gi_pref(node, ns, prefix, legacy, "termination.connect_success_every_k", p.connect_success_every_k);
    p.connect_success_dist_scale = gd_pref(node, ns, prefix, legacy, "termination.connect_success_dist_scale", p.connect_success_dist_scale);
    p.direct_connect_step_factor = gd_pref(node, ns, prefix, legacy, "termination.direct_connect_step_factor", p.direct_connect_step_factor);
    p.connect_target_tolerance = gd_pref(node, ns, prefix, legacy, "termination.connect_target_tolerance", p.connect_target_tolerance);

    p.aapf.enable = gb_pref(node, ns, prefix, legacy, "aapf.enable", p.aapf.enable);
    p.aapf.repulsion_range_m = gd_pref(
        node, ns, prefix, legacy, "aapf.repulsion_range_m", p.aapf.repulsion_range_m);
    p.aapf.stall_threshold_iters = gi_pref(
        node, ns, prefix, legacy,
        "aapf.stall_threshold_iters", p.aapf.stall_threshold_iters);
    p.aapf.strict_validation_distance = gd_pref(
        node, ns, prefix, legacy,
        "aapf.strict_validation_distance", p.aapf.strict_validation_distance);
    p.aapf.adaptive_window = gi_pref(
        node, ns, prefix, legacy, "aapf.adaptive_window", p.aapf.adaptive_window);
    p.aapf.adaptive_exploration = gd_pref(
        node, ns, prefix, legacy,
        "aapf.adaptive_exploration", p.aapf.adaptive_exploration);
    p.aapf.global_min_period = gi_pref(
        node, ns, prefix, legacy,
        "aapf.global_min_period", p.aapf.global_min_period);

    p.aapf.min_rewire_radius_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.min_rewire_radius_ratio", p.aapf.min_rewire_radius_ratio);
    p.aapf.shrink_motion_initial_scale = gd_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_initial_scale", p.aapf.shrink_motion_initial_scale);
    p.aapf.shrink_motion_decay = gd_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_decay", p.aapf.shrink_motion_decay);
    p.aapf.shrink_motion_attempts = gi_pref(
        node, ns, prefix, legacy,
        "aapf.shrink_motion_attempts", p.aapf.shrink_motion_attempts);
    p.aapf.bridge_node_sep_ratio = gd_pref(
        node, ns, prefix, legacy,
        "aapf.bridge_node_sep_ratio", p.aapf.bridge_node_sep_ratio);
    cfg.orientation.near_dist = gd_pref(node, ns, prefix, legacy, "orientation.near_dist", cfg.orientation.near_dist);
    cfg.orientation.ori_gate_dist = gd_pref(node, ns, prefix, legacy, "orientation.ori_gate_dist", cfg.orientation.ori_gate_dist);
    cfg.orientation.ori_far_tol_deg = gd_pref(node, ns, prefix, legacy, "orientation.ori_far_tol_deg", cfg.orientation.ori_far_tol_deg);
    cfg.orientation.ori_near_tol_deg = gd_pref(node, ns, prefix, legacy, "orientation.ori_near_tol_deg", cfg.orientation.ori_near_tol_deg);
    cfg.orientation.ori_weight_far = gd_pref(node, ns, prefix, legacy, "orientation.ori_weight_far", cfg.orientation.ori_weight_far);
    cfg.orientation.ori_weight_near = gd_pref(node, ns, prefix, legacy, "orientation.ori_weight_near", cfg.orientation.ori_weight_near);

    const auto fallback_values = gda_pref(node, ns, prefix, legacy, "fallback.levels", {});
    if (fallback_values.size() % 3U == 0U && !fallback_values.empty()) {
        cfg.orientation.fallback_levels.clear();
        for (size_t i = 0; i + 2U < fallback_values.size(); i += 3U) {
            cfg.orientation.fallback_levels.push_back(
                {fallback_values[i], fallback_values[i + 1U], fallback_values[i + 2U]});
        }
    }
    return cfg;
}

v2::PipelineOptions loadPipelineOptions(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    v2::PipelineOptions opts;
    opts.planner_config = loadPlannerConfig(node, ns);
    opts.ik_selector_params = loadIKSelectParams(node, ns);
    opts.analytical_ik_params = loadAnalyticalIKParams(node, ns);
    opts.planning_deadline_s = gd(
        node, ns, "fairino.planner.planning_deadline_s", opts.planning_deadline_s);
    opts.planning_deadline_s = gd(
        node, ns, "fairino.benchmark.planning_deadline_s", opts.planning_deadline_s);
    opts.goal_root_mode = gs(
        node, ns, "fairino.benchmark.goal_root_mode", opts.goal_root_mode);
    opts.anytime_checkpoints_s = gda(
        node, ns, "fairino.benchmark.anytime_checkpoints_s", opts.anytime_checkpoints_s);
    if (opts.goal_root_mode != "single_root" && opts.goal_root_mode != "multi_root") {
        throw std::invalid_argument(
            "fairino.benchmark.goal_root_mode must be single_root or multi_root");
    }
    if (!(opts.planning_deadline_s > 0.0)) {
        throw std::invalid_argument("Fairino planning deadline must be positive");
    }
    opts.enable_path_optimizer = gb(node, ns, "fairino.planner.enable_path_optimizer", true);
    opts.optimizer_fail_open_return_original = gb(
        node, ns, "fairino.planner.optimizer_fail_open_return_original", opts.optimizer_fail_open_return_original);
    opts.use_multi_obstacle_input = gb(
        node, ns, "fairino.planner.use_multi_obstacle_input", opts.use_multi_obstacle_input);
    opts.min_obstacle_size_threshold = gd(
        node, ns, "fairino.planner.min_obstacle_size_threshold", opts.min_obstacle_size_threshold);
    opts.optimizer_validation_distance = gd(
        node, ns, "fairino.optimizer.validation_distance", opts.optimizer_validation_distance);
    opts.optimizer_shortcut_trials = gi(
        node, ns, "fairino.optimizer.shortcut_trials", opts.optimizer_shortcut_trials);
    opts.optimizer_pull_trials = gi(
        node, ns, "fairino.optimizer.pull_trials", opts.optimizer_pull_trials);
    opts.optimizer_densify_max_spacing = gd(
        node, ns, "fairino.optimizer.densify_max_spacing", opts.optimizer_densify_max_spacing);
    opts.optimizer_pull_alpha_min = gd(
        node, ns, "fairino.optimizer.pull_alpha_min", opts.optimizer_pull_alpha_min);
    opts.optimizer_pull_alpha_max = gd(
        node, ns, "fairino.optimizer.pull_alpha_max", opts.optimizer_pull_alpha_max);
    opts.optimizer_orientation_check_count = gi(
        node, ns, "fairino.optimizer.orientation_check_count", opts.optimizer_orientation_check_count);
    opts.final_validation_distance = gd(
        node, ns, "fairino.safety.final_validation_distance", opts.final_validation_distance);
    opts.final_validation_fail_open = gb(
        node, ns, "fairino.planner.final_validation_fail_open", opts.final_validation_fail_open);
    opts.trajectory_waypoint_dt = gd(
        node, ns, "fairino.trajectory.waypoint_dt", opts.trajectory_waypoint_dt);
    opts.planner_random_seed = static_cast<unsigned int>(std::max(
        0, gi(node, ns, "fairino.planner.random_seed", static_cast<int>(opts.planner_random_seed))));
    opts.default_obstacle_origin = vector3From(
        gda(node, ns, "fairino.pipeline.default_obstacle_origin",
            {opts.default_obstacle_origin.x(), opts.default_obstacle_origin.y(), opts.default_obstacle_origin.z()}),
        opts.default_obstacle_origin);
    opts.default_obstacle_size = vector3From(
        gda(node, ns, "fairino.pipeline.default_obstacle_size",
            {opts.default_obstacle_size.x(), opts.default_obstacle_size.y(), opts.default_obstacle_size.z()}),
        opts.default_obstacle_size);
    return opts;
}

std::string loadToolModelOverride(const rclcpp::Node::SharedPtr& node, const std::string& ns) {
    return gs(node, ns, "fairino.ik.tool_model_override", "auto");
}

}  // namespace fairino_planning::config
