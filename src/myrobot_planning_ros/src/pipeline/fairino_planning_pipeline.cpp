#include "myrobot_planning_ros/pipeline/fairino_planning_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometric_shapes/shapes.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "myrobot_planning_core/constraints/orientation_checker.h"
#include "myrobot_planning_core/dh_kinematics.h"
#include "myrobot_planning_core/engine/planner_engine.hpp"
#include "myrobot_planning_core/trajectory/path_shortcut.h"
#include "myrobot_planning_core/trajectory/trajectory_smoother.h"
#include "myrobot_planning_ros/config/moveit_error_mapping.h"
#include "myrobot_planning_ros/moveit_collision_checker.h"

namespace fairino_planning::v2 {

namespace {

class CountingCollisionChecker final : public CollisionInterface {
public:
    explicit CountingCollisionChecker(std::shared_ptr<CollisionInterface> delegate)
        : delegate_(std::move(delegate)) {}

    bool isStateValid(const JointConfig& q) const override {
        ++state_checks_;
        return delegate_->isStateValid(q);
    }

    bool isMotionValid(const JointConfig& from, const JointConfig& to,
                       double validation_distance) const override {
        ++motion_checks_;
        const bool valid = delegate_->isMotionValid(from, to, validation_distance);
        if (valid) ++valid_motion_edges_;
        else ++invalid_motion_edges_;
        return valid;
    }

    bool supportsWorldClearance() const override {
        return delegate_->supportsWorldClearance();
    }

    ClearanceQueryResult nearestWorldClearance(
        const JointConfig& q, double influence_distance) const override {
        ++clearance_checks_;
        return delegate_->nearestWorldClearance(q, influence_distance);
    }

    int stateChecks() const { return state_checks_; }
    int motionChecks() const { return motion_checks_; }
    int validMotionEdges() const { return valid_motion_edges_; }
    int invalidMotionEdges() const { return invalid_motion_edges_; }
    int clearanceChecks() const { return clearance_checks_; }

private:
    std::shared_ptr<CollisionInterface> delegate_;
    mutable int state_checks_ = 0;
    mutable int motion_checks_ = 0;
    mutable int valid_motion_edges_ = 0;
    mutable int invalid_motion_edges_ = 0;
    mutable int clearance_checks_ = 0;
};

void appendAnytimeTrace(const PlanResult& result, unsigned int planner_seed) {
    const char* configured_path = std::getenv("FAIRINO_ANYTIME_TRACE_PATH");
    if (!configured_path || *configured_path == '\0') return;
    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);
    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) return;
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,checkpoint_s,has_solution,incumbent_joint_cost_rad,"
               << "cumulative_iterations,cumulative_sample_attempts,cumulative_accepted_samples,"
               << "cumulative_nodes,cumulative_work_units\n";
    }
    ++run_index;
    for (const auto& point : result.anytime_trace) {
        stream << run_index << ',' << planner_seed << ',' << point.elapsed_s << ','
               << (point.has_solution ? "true" : "false") << ','
               << point.incumbent_joint_cost_rad << ',' << point.iterations << ','
               << point.sample_attempts << ',' << point.accepted_samples << ','
               << point.num_nodes << ',' << point.work_units << '\n';
    }
}

void appendRootDiagnostics(
    const std::vector<JointConfig>& roots,
    const std::vector<IKCandidateDiagnostic>& diagnostics,
    const JointConfig& start,
    const PlanResult& result,
    unsigned int planner_seed) {
    const char* configured_path = std::getenv("FAIRINO_ROOT_DIAGNOSTICS_PATH");
    if (!configured_path || *configured_path == 0) return;
    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);
    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) return;
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,root_index,q1,q2,q3,q4,q5,q6,"
               << "passed_hard_filter,filter_reason,total_cost,assigned_sample_attempts,"
               << "accepted_samples,path_improvements,selected_final\n";
    }
    ++run_index;
    std::vector<IKCandidateDiagnostic> records = diagnostics;
    if (records.empty()) {
        for (const auto& root : roots) {
            IKCandidateDiagnostic record;
            record.q = root;
            record.passed_hard_filter = true;
            record.reject_reason = IKRejectReason::kAccepted;
            record.total_cost = (root - start).norm();
            records.push_back(record);
        }
    }
    for (size_t index = 0U; index < records.size(); ++index) {
        const auto& record = records[index];
        int planning_root = -1;
        for (size_t root_index = 0U; root_index < roots.size(); ++root_index) {
            if ((record.q - roots[root_index]).norm() < 1e-8) {
                planning_root = static_cast<int>(root_index);
                break;
            }
        }
        bool selected = false;
        if (result.success && !result.path.empty()) {
            selected = (result.path.back() - record.q).norm() < 1e-6;
        }
        const auto effort_at = [planning_root](const std::vector<int>& values) {
            return planning_root >= 0 && static_cast<size_t>(planning_root) < values.size()
                ? values[static_cast<size_t>(planning_root)] : 0;
        };
        stream << run_index << "," << planner_seed << "," << index;
        for (int joint = 0; joint < NUM_JOINTS; ++joint) stream << "," << record.q[joint];
        const bool accepted_for_planning = planning_root >= 0;
        const std::string filter_reason = accepted_for_planning
            ? "" : (record.passed_hard_filter ? "not_selected_for_planning"
                                                   : toString(record.reject_reason));
        stream << "," << (accepted_for_planning ? "true" : "false") << ","
               << filter_reason << ","
               << record.total_cost << ","
               << effort_at(result.effort_stats.root_sample_attempts) << ","
               << effort_at(result.effort_stats.root_accepted_samples) << ","
               << effort_at(result.effort_stats.root_path_improvements) << ","
               << (selected ? "true" : "false") << "\n";
    }
}

void appendTrajectoryPaths(
    const std::vector<JointConfig>& raw_path,
    const std::vector<JointConfig>& final_path,
    const Transform4d& flange_to_tool,
    ToolModel tool_model,
    unsigned int planner_seed) {
    const char* configured_path = std::getenv("FAIRINO_TRAJECTORY_PATHS_PATH");
    if (!configured_path || *configured_path == '\0') return;
    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);
    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) return;
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,path_stage,waypoint_index,q1,q2,q3,q4,q5,q6,tcp_x,tcp_y,tcp_z\n";
    }
    ++run_index;
    DHKinematics fk;
    fk.setToolTransform(flange_to_tool);
    const auto write_path = [&](const char* stage, const std::vector<JointConfig>& values) {
        for (size_t index = 0U; index < values.size(); ++index) {
            const Vector3d tcp = fk.fkine(values[index], tool_model).block<3, 1>(0, 3);
            stream << run_index << ',' << planner_seed << ',' << stage << ',' << index;
            for (int joint = 0; joint < NUM_JOINTS; ++joint) stream << ',' << values[index][joint];
            stream << ',' << tcp.x() << ',' << tcp.y() << ',' << tcp.z() << '\n';
        }
    };
    write_path("raw", raw_path);
    write_path("final", final_path);
}

void appendSamplingStats(
    const PlanResult& result,
    unsigned int planner_seed) {
    const char* configured_path = std::getenv("FAIRINO_AAPF_STATS_PATH");
    if (!configured_path || *configured_path == '\0' || result.sampling_stats.empty()) {
        return;
    }

    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);

    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(
        path,
        std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) {
        return;
    }
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,tree,source,selections,proposals,insertions,"
               << "progress_events,connections,improvements,clearance_queries,"
               << "utility_sum,elapsed_ms\n";
    }
    ++run_index;
    for (const auto& stats : result.sampling_stats) {
        stream << run_index << ',' << planner_seed << ',' << stats.tree_index << ','
               << stats.source << ',' << stats.selections << ',' << stats.proposals << ','
               << stats.insertions << ',' << stats.progress_events << ',' << stats.connections
               << ',' << stats.improvements << ',' << stats.clearance_queries << ','
               << stats.utility_sum << ',' << stats.elapsed_ms << '\n';
    }
}

void appendMireStats(
    const PlanResult& result,
    unsigned int planner_seed) {
    const char* configured_path = std::getenv("FAIRINO_MIRE_STATS_PATH");
    if (!configured_path || *configured_path == '\0') {
        return;
    }

    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);
    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) {
        return;
    }
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,batches,work_units,sample_attempts,accepted_samples,"
               << "uniform_sample_attempts,informed_sample_attempts,"
               << "uniform_accepted_samples,informed_accepted_samples,vertices,edges,"
               << "reverse_queue_pops,reverse_queue_stale_discards,reverse_queue_consistent_discards,"
               << "forward_edge_pops,goal_tree_edge_pops,"
               << "sparse_state_checks,edge_validation_calls,"
               << "final_validation_calls,state_validation_calls,blocked_edges,selected_goal_root,"
               << "focal_entries_examined,goal_root_count,"
               << "budget_exhausted,first_solution_time_s,first_solution_path_cost_rad,lower_bound_certified,"
               << "post_solution_sample_attempts,post_solution_budget_complete,"
               << "incumbent_improvements,informed_path_improvements,"
               << "post_cost_search_pops,post_candidate_paths,post_strict_improvements,"
               << "sparse_rejected_edges,quarter_state_checks,full_invalid_edges_avoided,"
               << "selected_goal_root_before,selected_goal_root_after,"
               << "joint_cost_reduction_rad,tcp_cost_reduction_m,"
               << "root_lower_bounds_rad,root_sample_attempts,root_accepted_samples,root_path_improvements,"
               << "raw_path_cost_rad\n";
    }
    const auto& stats = result.effort_stats;
    const auto join = [](const auto& values) {
        std::ostringstream output;
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0U) output << ';';
            output << values[i];
        }
        return output.str();
    };
    stream << ++run_index << ',' << planner_seed << ',' << stats.batches << ','
           << stats.work_units << ',' << stats.sample_attempts << ',' << stats.sampled_states << ','
           << stats.uniform_sample_attempts << ',' << stats.informed_sample_attempts << ','
           << stats.uniform_accepted_states << ',' << stats.informed_accepted_states << ','
           << result.num_nodes << ',' << stats.graph_edges << ','
           << stats.reverse_queue_pops << ',' << stats.reverse_queue_stale_discards << ','
           << stats.reverse_queue_consistent_discards << ',' << stats.forward_edge_pops << ','
           << stats.goal_tree_edge_pops << ','
           << stats.sparse_state_checks << ',' << stats.edge_validation_attempts << ','
           << stats.final_validation_calls << ',' << stats.state_validation_calls << ','
           << stats.blocked_edges << ',' << stats.selected_goal_root << ','
           << stats.focal_entries_examined << ','
           << stats.goal_root_count << ','
           << (stats.budget_exhausted ? "true" : "false") << ','
           << stats.first_solution_time_s << ','
           << stats.first_solution_path_cost << ','
           << (stats.lower_bound_certified ? "true" : "false") << ','
           << stats.post_solution_sample_attempts << ','
           << (stats.post_solution_budget_complete ? "true" : "false") << ','
           << stats.incumbent_improvements << ',' << stats.informed_path_improvements << ','
           << stats.post_cost_search_pops << ',' << stats.post_candidate_paths << ','
           << stats.post_strict_improvements << ',' << stats.sparse_rejected_edges << ','
           << stats.quarter_state_checks << ',' << stats.full_invalid_edges_avoided << ','
           << stats.selected_goal_root_before << ',' << stats.selected_goal_root_after << ','
           << stats.joint_cost_reduction_rad << ',' << stats.tcp_cost_reduction_m << ','
           << join(stats.root_lower_bounds) << ',' << join(stats.root_sample_attempts) << ','
           << join(stats.root_accepted_samples) << ',' << join(stats.root_path_improvements) << ','
           << result.path_cost << '\n';
}

void appendPlannerDiagnostics(
    const PlanResult& result,
    unsigned int planner_seed,
    const std::string& planner_id,
    bool plan_success,
    double raw_joint_path_cost_rad,
    double optimized_joint_path_length_rad,
    double optimized_tcp_path_length_m,
    double path_optimizer_time_s = 0.0,
    double joint_turn_total_variation_rad = std::numeric_limits<double>::quiet_NaN(),
    int waypoint_count = 0,
    double ik_time_s = 0.0,
    double root_generation_time_s = 0.0,
    double final_validation_time_s = 0.0,
    double trajectory_construction_time_s = 0.0) {
    const char* configured_path = std::getenv("FAIRINO_PLANNER_DIAGNOSTICS_PATH");
    if (!configured_path || *configured_path == '\0') return;

    static std::mutex write_mutex;
    static std::string active_path;
    static int run_index = 0;
    std::lock_guard<std::mutex> lock(write_mutex);
    const std::string path(configured_path);
    const bool new_path = path != active_path || !std::filesystem::exists(path);
    std::ofstream stream(path, std::ios::out | (new_path ? std::ios::trunc : std::ios::app));
    if (!stream) return;
    if (new_path) {
        active_path = path;
        run_index = 0;
        stream << "run_index,planner_seed,planner_id,plan_success,stop_reason,iterations,sample_attempts,"
               << "accepted_samples,num_nodes,work_units,graph_edges,optimized_joint_path_length_rad,"
               << "optimized_tcp_path_length_m,raw_joint_path_cost_rad,"
               << "joint_turn_total_variation_rad,waypoint_count,"
               << "first_solution_time_s,first_solution_path_cost_rad,lower_bound_certified,"
               << "post_solution_sample_attempts,post_solution_budget_complete,"
               << "collision_state_checks,collision_motion_checks,valid_motion_edges,invalid_motion_edges,"
               << "valid_motion_edge_efficiency,collision_clearance_checks,"
               << "path_optimizer_time_s,ik_time_s,root_generation_time_s,search_time_s,"
               << "final_validation_time_s,trajectory_construction_time_s,goal_root_count,selected_goal_root,"
               << "total_planning_time_s\n";
    }
    stream << ++run_index << ',' << planner_seed << ',' << planner_id << ','
           << (plan_success ? "true" : "false") << ',' << result.stop_reason << ','
           << result.iterations << ',' << result.sample_attempts << ','
           << result.accepted_samples << ',' << result.num_nodes << ','
           << result.effort_stats.work_units << ',' << result.effort_stats.graph_edges << ','
           << optimized_joint_path_length_rad << ',' << optimized_tcp_path_length_m << ','
           << raw_joint_path_cost_rad << ',' << joint_turn_total_variation_rad << ','
           << waypoint_count << ','
           << result.first_solution_time_s << ','
           << result.first_solution_path_cost << ','
           << (result.lower_bound_certified ? "true" : "false") << ','
           << result.effort_stats.post_solution_sample_attempts << ','
           << (result.effort_stats.post_solution_budget_complete ? "true" : "false") << ','
           << result.collision_state_checks << ',' << result.collision_motion_checks << ','
           << result.valid_motion_edges << ',' << result.invalid_motion_edges << ','
           << (result.collision_motion_checks > 0
                   ? static_cast<double>(result.valid_motion_edges) / result.collision_motion_checks
                   : 0.0) << ',' << result.collision_clearance_checks << ',' << path_optimizer_time_s << ','
           << ik_time_s << ',' << root_generation_time_s << ',' << result.planning_time << ','
           << final_validation_time_s << ',' << trajectory_construction_time_s << ','
           << result.effort_stats.goal_root_count << ','
           << result.effort_stats.selected_goal_root << ','
           << ik_time_s + root_generation_time_s + result.planning_time +
                  path_optimizer_time_s + final_validation_time_s + trajectory_construction_time_s << '\n';
}

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

double jointPathLength(const std::vector<JointConfig>& path) {
    if (path.size() < 2U) {
        return 0.0;
    }
    double length = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        // The trajectory controller receives these bounded joint values
        // directly, so quality reporting must use the executed displacement.
        length += (path[i] - path[i - 1U]).norm();
    }
    return length;
}

double tcpPathLength(
    const std::vector<JointConfig>& path,
    const Transform4d& flange_to_tool,
    ToolModel tool_model) {
    if (path.size() < 2U) return 0.0;
    DHKinematics fk;
    fk.setToolTransform(flange_to_tool);
    Vector3d previous = fk.fkine(path.front(), tool_model).block<3, 1>(0, 3);
    double length = 0.0;
    for (size_t i = 1U; i < path.size(); ++i) {
        const Vector3d current = fk.fkine(path[i], tool_model).block<3, 1>(0, 3);
        length += (current - previous).norm();
        previous = current;
    }
    return length;
}

double jointTurnVariation(const std::vector<JointConfig>& path) {
    double variation = 0.0;
    for (size_t i = 2U; i < path.size(); ++i) {
        const JointConfig before = path[i - 1U] - path[i - 2U];
        const JointConfig after = path[i] - path[i - 1U];
        if (before.norm() <= 1e-12 || after.norm() <= 1e-12) continue;
        const double cosine = std::clamp(
            before.dot(after) / (before.norm() * after.norm()), -1.0, 1.0);
        variation += std::acos(cosine);
    }
    return variation;
}

std::string goalTipLink(const moveit_msgs::msg::Constraints& constraints) {
    if (!constraints.orientation_constraints.empty() &&
        !constraints.orientation_constraints[0].link_name.empty()) {
        return constraints.orientation_constraints[0].link_name;
    }
    if (!constraints.position_constraints.empty() &&
        !constraints.position_constraints[0].link_name.empty()) {
        return constraints.position_constraints[0].link_name;
    }
    return {};
}

bool copyJointGroupToConfig(
    const moveit::core::RobotState& state,
    const moveit::core::JointModelGroup* jmg,
    JointConfig& out) {
    if (!jmg) {
        return false;
    }
    std::vector<double> values;
    state.copyJointGroupPositions(jmg, values);
    if (values.empty()) {
        return false;
    }

    out = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(values.size()); ++i) {
        out[i] = values[i];
    }
    return true;
}

std::vector<JointConfig> collectGlobalGoalCandidates(
    const JointConfig& preferred_goal,
    bool pose_goal,
    const Transform4d& requested_pose,
    ToolModel tool_model,
    const PipelineOptions& options,
    const Transform4d& flange_to_tool,
    const MoveItCollisionChecker& collision,
    std::vector<IKCandidateDiagnostic>* all_diagnostics) {
    std::vector<JointConfig> candidates;
    if (all_diagnostics) all_diagnostics->clear();
    const JointLimits limits;
    const auto append = [&](const JointConfig& q) {
        if (!limits.isWithin(q) || !collision.isStateValid(q)) return;
        for (const auto& existing : candidates) {
            if ((q - existing).norm() < 1e-9) return;
        }
        candidates.push_back(q);
    };

    append(preferred_goal);
    if (!pose_goal || options.ik_selector_params.task_profile != IKTaskProfile::Continuous) {
        return candidates;
    }

    FairinoIK ik(options.analytical_ik_params);
    IKSelector selector(options.ik_selector_params);
    ik.setToolTransform(flange_to_tool);
    selector.setToolTransform(flange_to_tool);
    const auto ik_result = ik.solve(requested_pose, tool_model);
    if (!ik_result.success) return candidates;

    IKSelectionRequest select_request;
    select_request.solutions = &ik_result.solutions;
    select_request.seed = preferred_goal;
    select_request.target_pose = requested_pose;
    select_request.tool_model = tool_model;
    select_request.task_profile = IKTaskProfile::Continuous;
    auto selection = selector.select(select_request);
    if (all_diagnostics) {
        *all_diagnostics = selection.diagnostics;
        const bool preferred_listed = std::any_of(
            all_diagnostics->begin(), all_diagnostics->end(), [&](const auto& item) {
                return (item.q - preferred_goal).norm() < 1e-8;
            });
        if (!preferred_listed) {
            IKCandidateDiagnostic preferred;
            preferred.q = preferred_goal;
            preferred.passed_hard_filter = !candidates.empty();
            preferred.reject_reason = preferred.passed_hard_filter
                ? IKRejectReason::kAccepted : IKRejectReason::kOutsideLimits;
            preferred.total_cost = (preferred_goal - select_request.seed).norm();
            all_diagnostics->push_back(preferred);
        }
    }
    std::vector<IKCandidateDiagnostic> accepted;
    for (const auto& diagnostic : selection.diagnostics) {
        if (diagnostic.passed_hard_filter) accepted.push_back(diagnostic);
    }
    std::sort(accepted.begin(), accepted.end(), [&](const auto& a, const auto& b) {
        if (a.selected != b.selected) return a.selected;
        if (std::abs(a.max_abs_dq - b.max_abs_dq) > 1e-9) {
            return a.max_abs_dq < b.max_abs_dq;
        }
        if (std::abs(a.dq_norm - b.dq_norm) > 1e-9) return a.dq_norm < b.dq_norm;
        return a.total_cost < b.total_cost;
    });

    const int max_roots = std::max(1, options.ik_selector_params.continuous_max_goal_roots);
    const double min_separation = std::max(
        0.0, options.ik_selector_params.continuous_goal_root_min_separation_rad);
    for (const auto& diagnostic : accepted) {
        if (static_cast<int>(candidates.size()) >= max_roots) break;
        bool diverse = true;
        for (const auto& existing : candidates) {
            if ((diagnostic.q - existing).norm() < min_separation) {
                diverse = false;
                break;
            }
        }
        if (diverse) append(diagnostic.q);
    }
    for (const auto& diagnostic : accepted) {
        if (static_cast<int>(candidates.size()) >= max_roots) break;
        append(diagnostic.q);
    }
    return candidates;
}

bool validateJointPath(
    const std::vector<JointConfig>& path,
    const std::shared_ptr<MoveItCollisionChecker>& collision,
    double validation_distance,
    int* invalid_segment) {
    if (invalid_segment) {
        *invalid_segment = -1;
    }
    if (path.empty() || !collision) {
        return false;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        if (!collision->isStateValid(path[i])) {
            if (invalid_segment) {
                *invalid_segment = static_cast<int>(i);
            }
            return false;
        }
        if (i > 0U &&
            !collision->isMotionValid(path[i - 1U], path[i], validation_distance)) {
            if (invalid_segment) {
                *invalid_segment = static_cast<int>(i - 1U);
            }
            return false;
        }
    }
    return true;
}

std::vector<JointConfig> decimatePathForExport(
    const std::vector<JointConfig>& path,
    const std::shared_ptr<MoveItCollisionChecker>& collision,
    double validation_distance) {
    if (path.size() <= 2U || !collision) {
        return path;
    }

    const double max_export_segment = std::max(0.02, validation_distance * 2.0);
    std::vector<JointConfig> out;
    out.reserve(path.size());
    size_t i = 0;
    out.push_back(path.front());
    while (i + 1U < path.size()) {
        size_t best = i + 1U;
        for (size_t j = path.size() - 1U; j > i + 1U; --j) {
            const double segment_len = (path[j] - path[i]).norm();
            if (segment_len <= max_export_segment &&
                collision->isStateValid(path[j]) &&
                collision->isMotionValid(path[i], path[j], validation_distance)) {
                best = j;
                break;
            }
        }
        out.push_back(path[best]);
        i = best;
    }
    return out;
}

std::vector<JointConfig> densifyPathForExecution(
    const std::vector<JointConfig>& path, double max_spacing) {
    if (path.size() < 2U || max_spacing <= 0.0 || !std::isfinite(max_spacing)) {
        return path;
    }

    std::vector<JointConfig> dense;
    dense.reserve(path.size());
    dense.push_back(path.front());
    for (size_t i = 1; i < path.size(); ++i) {
        const JointConfig delta = path[i] - path[i - 1U];
        const int segments = std::max(
            1, static_cast<int>(std::ceil(delta.norm() / max_spacing)));
        for (int segment = 1; segment <= segments; ++segment) {
            const double alpha = static_cast<double>(segment) / segments;
            dense.push_back(path[i - 1U] + alpha * delta);
        }
    }
    return dense;
}

}  // namespace

bool FairinoPlanningPipeline::solve(
    const planning_scene::PlanningSceneConstPtr& scene,
    const moveit_msgs::msg::MotionPlanRequest& req,
    const std::string& group_name,
    const std::shared_ptr<PlanningAlgorithm>& algorithm,
    const PipelineOptions& options,
    planning_interface::MotionPlanResponse& res) const {
    const double configured_deadline_s = std::max(1e-3, options.planning_deadline_s);
    const double requested_deadline_s = req.allowed_planning_time > 0.0
        ? req.allowed_planning_time
        : configured_deadline_s;
    const double effective_deadline_s = std::min(configured_deadline_s, requested_deadline_s);
    RCLCPP_INFO(
        logger_,
        "Fairino planning deadline: request=%.3fs configured=%.3fs effective=%.3fs",
        req.allowed_planning_time, configured_deadline_s, effective_deadline_s);
    const auto request_started = std::chrono::steady_clock::now();
    const auto request_deadline = request_started + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(effective_deadline_s));
    const auto request_should_stop = [&]() {
        return (options.cancel_requested && options.cancel_requested()) ||
            std::chrono::steady_clock::now() >= request_deadline;
    };
    if (!scene) {
        RCLCPP_ERROR(logger_, "PlanningScene is null in FairinoPlanningPipeline::solve()");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return false;
    }

    const auto* jmg = scene->getRobotModel()->getJointModelGroup(group_name);
    if (!jmg) {
        RCLCPP_ERROR(logger_, "Joint model group '%s' not found", group_name.c_str());
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
        return false;
    }

    if (!algorithm) {
        RCLCPP_ERROR(logger_, "Algorithm instance is null");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return false;
    }

    Transform4d flange_to_tool = Transform4d::Identity();
    if (!loadFlangeToTool(*scene->getRobotModel(), flange_to_tool)) {
        RCLCPP_ERROR(logger_,
            "Fairino planning requires a fixed wrist3_link -> tool0 TCP chain in robot_description.");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
        return false;
    }
    algorithm->setToolTransform(flange_to_tool);
    const ToolModel tool_model = ToolModel::GRIPPER;

    moveit::core::RobotState scene_start_state(scene->getRobotModel());
    scene_start_state = scene->getCurrentState();
    moveit::core::RobotState start_state = scene_start_state;
    if (!req.start_state.joint_state.name.empty()) {
        moveit::core::RobotState requested_start_state(scene->getRobotModel());
        requested_start_state = scene_start_state;
        moveit::core::robotStateMsgToRobotState(req.start_state, requested_start_state);

        std::vector<double> scene_start_vals;
        std::vector<double> requested_start_vals;
        scene_start_state.copyJointGroupPositions(jmg, scene_start_vals);
        requested_start_state.copyJointGroupPositions(jmg, requested_start_vals);
        JointConfig q_scene_start = JointConfig::Zero();
        JointConfig q_requested_start = JointConfig::Zero();
        for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(scene_start_vals.size()); ++i) {
            q_scene_start[i] = scene_start_vals[i];
        }
        for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(requested_start_vals.size()); ++i) {
            q_requested_start[i] = requested_start_vals[i];
        }
        const double start_state_delta = wrapToPi(q_requested_start - q_scene_start).norm();
        if (start_state_delta <= 0.20) {
            start_state = requested_start_state;
        } else {
            RCLCPP_WARN(
                logger_,
                "Ignoring request start_state because it differs from PlanningScene current state by %.6f rad",
                start_state_delta);
        }
    }

    std::vector<double> start_vals;
    start_state.copyJointGroupPositions(jmg, start_vals);
    JointConfig q_start = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(start_vals.size()); ++i) {
        q_start[i] = start_vals[i];
    }
    std::vector<double> scene_current_vals;
    scene_start_state.copyJointGroupPositions(jmg, scene_current_vals);
    JointConfig q_scene_current = JointConfig::Zero();
    for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(scene_current_vals.size()); ++i) {
        q_scene_current[i] = scene_current_vals[i];
    }

    JointConfig q_goal = JointConfig::Zero();
    bool goal_found = false;
    bool pose_goal = false;
    Transform4d requested_pose_goal = Transform4d::Identity();

    if (!req.goal_constraints.empty()) {
        const auto& gc = req.goal_constraints[0];

        if (!gc.joint_constraints.empty()) {
            const auto& joint_names = jmg->getActiveJointModelNames();
            q_goal = q_start;
            for (const auto& jc : gc.joint_constraints) {
                for (int i = 0; i < NUM_JOINTS && i < static_cast<int>(joint_names.size()); ++i) {
                    if (jc.joint_name == joint_names[i]) {
                        q_goal[i] = jc.position;
                        break;
                    }
                }
            }
            goal_found = true;
        }

        if (!goal_found &&
            (!gc.position_constraints.empty() || !gc.orientation_constraints.empty())) {
            geometry_msgs::msg::Pose target_pose;
            bool has_pos = false, has_ori = false;
            if (!gc.position_constraints.empty()) {
                const auto& pc = gc.position_constraints[0];
                if (!pc.constraint_region.primitive_poses.empty()) {
                    target_pose.position.x = pc.constraint_region.primitive_poses[0].position.x;
                    target_pose.position.y = pc.constraint_region.primitive_poses[0].position.y;
                    target_pose.position.z = pc.constraint_region.primitive_poses[0].position.z;
                    has_pos = true;
                }
            }
            if (!gc.orientation_constraints.empty()) {
                target_pose.orientation = gc.orientation_constraints[0].orientation;
                has_ori = true;
            }

            if (has_pos && has_ori) {
                const std::string tip_link = goalTipLink(gc);
                moveit::core::RobotState ik_state = start_state;
                ik_state.update();

                bool ik_ok = false;
                try {
                    if (!tip_link.empty()) {
                        ik_ok = ik_state.setFromIK(jmg, target_pose, tip_link, 0.2);
                    } else {
                        ik_ok = ik_state.setFromIK(jmg, target_pose, 0.2);
                    }
                } catch (const std::exception& ex) {
                    RCLCPP_WARN(
                        logger_,
                        "MoveIt IK threw for pose goal: group=%s tip=%s error=%s",
                        group_name.c_str(),
                        tip_link.empty() ? "<default>" : tip_link.c_str(),
                        ex.what());
                }

                if (ik_ok && copyJointGroupToConfig(ik_state, jmg, q_goal)) {
                    goal_found = true;
                    pose_goal = true;
                    const Eigen::Quaterniond orientation(
                        target_pose.orientation.w, target_pose.orientation.x,
                        target_pose.orientation.y, target_pose.orientation.z);
                    requested_pose_goal.block<3, 3>(0, 0) = orientation.normalized().toRotationMatrix();
                    requested_pose_goal.block<3, 1>(0, 3) = Eigen::Vector3d(
                        target_pose.position.x, target_pose.position.y, target_pose.position.z);
                    RCLCPP_INFO(
                        logger_,
                        "MoveIt IK candidate accepted: group=%s tip=%s",
                        group_name.c_str(),
                        tip_link.empty() ? "<default>" : tip_link.c_str());
                } else {
                    RCLCPP_WARN(
                        logger_,
                        "MoveIt IK failed for pose goal: group=%s tip=%s",
                        group_name.c_str(),
                        tip_link.empty() ? "<default>" : tip_link.c_str());
                }
            }
        }
    }

    if (!goal_found) {
        RCLCPP_ERROR(logger_, "No valid goal constraints found.");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
        return false;
    }

    const double ik_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - request_started).count();

    const double final_validation_distance =
        std::max(1e-4, options.final_validation_distance);
    PlannerConfig effective_planner_config = options.planner_config;
    if (effective_planner_config.planning.validation_distance >
        final_validation_distance) {
        RCLCPP_INFO(
            logger_,
            "PlannerValidationClamp: planner_validation_distance=%.4f final_validation_distance=%.4f",
            effective_planner_config.planning.validation_distance,
            final_validation_distance);
        effective_planner_config.planning.validation_distance =
            final_validation_distance;
    }

    auto collision = std::make_shared<MoveItCollisionChecker>(scene, group_name);
    auto planning_collision = std::make_shared<CountingCollisionChecker>(collision);
    PlannerEngine engine(algorithm);
    engine.setCollisionChecker(planning_collision);
    engine.configure(effective_planner_config);

    DHKinematics fk(DHParams{}, flange_to_tool);
    const auto T_start = fk.fkine(q_start, tool_model);
    const auto T_goal = pose_goal ? requested_pose_goal : fk.fkine(q_goal, tool_model);
    const Vector3d p_start = T_start.block<3, 1>(0, 3);
    const Vector3d p_goal = T_goal.block<3, 1>(0, 3);
    const RotMatrix3d R_target = T_goal.block<3, 3>(0, 0);
    const auto root_generation_started = std::chrono::steady_clock::now();
    std::vector<JointConfig> goal_candidates;
    std::vector<IKCandidateDiagnostic> goal_root_diagnostics;
    if (options.goal_root_mode == "single_root") {
        const JointLimits limits;
        if (limits.isWithin(q_goal) && collision->isStateValid(q_goal)) {
            goal_candidates.push_back(q_goal);
            IKCandidateDiagnostic diagnostic;
            diagnostic.q = q_goal;
            diagnostic.passed_hard_filter = true;
            diagnostic.reject_reason = IKRejectReason::kAccepted;
            diagnostic.total_cost = (q_goal - q_start).norm();
            diagnostic.selected = true;
            goal_root_diagnostics.push_back(diagnostic);
        }
    } else {
        goal_candidates = collectGlobalGoalCandidates(
            q_goal, pose_goal, T_goal, tool_model, options, flange_to_tool, *collision, &goal_root_diagnostics);
    }
    const double root_generation_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - root_generation_started).count();
    if (goal_candidates.empty()) {
        RCLCPP_ERROR(logger_, "No collision-free goal IK root is available.");
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::GOAL_IN_COLLISION;
        return false;
    }

    Vector3d obs_origin = options.default_obstacle_origin;
    Vector3d obs_size = options.default_obstacle_size;
    std::vector<ObstacleInfo> obstacles;
    size_t filtered_obstacles = 0;

    if (options.use_multi_obstacle_input) {
        const auto& collision_objects = scene->getWorld()->getObjectIds();
        obstacles.reserve(collision_objects.size());
        for (const auto& obj_id : collision_objects) {
            const auto obj = scene->getWorld()->getObject(obj_id);
            if (!obj || obj->shapes_.empty() || obj->shape_poses_.empty()) {
                ++filtered_obstacles;
                continue;
            }

            bool object_added = false;
            const size_t shape_count = std::min(obj->shapes_.size(), obj->shape_poses_.size());
            for (size_t i = 0; i < shape_count; ++i) {
                const auto* shape_raw = obj->shapes_[i].get();
                Eigen::Vector3d size(Eigen::Vector3d::Zero());
                const char* shape_type = "unknown";

                if (const auto* box = dynamic_cast<const shapes::Box*>(shape_raw)) {
                    size = Eigen::Vector3d(box->size[0], box->size[1], box->size[2]);
                    shape_type = "box";
                } else if (const auto* sphere = dynamic_cast<const shapes::Sphere*>(shape_raw)) {
                    const double d = 2.0 * sphere->radius;
                    size = Eigen::Vector3d(d, d, d);
                    shape_type = "sphere";
                } else if (const auto* cylinder = dynamic_cast<const shapes::Cylinder*>(shape_raw)) {
                    const double d = 2.0 * cylinder->radius;
                    size = Eigen::Vector3d(d, d, cylinder->length);
                    shape_type = "cylinder";
                } else {
                    ++filtered_obstacles;
                    continue;
                }

                if (size.minCoeff() < options.min_obstacle_size_threshold) {
                    RCLCPP_DEBUG(logger_,
                        "  obstacle '%s' shape[%zu] type=%s filtered (size too small: %.4f)",
                        obj_id.c_str(), i, shape_type, size.minCoeff());
                    ++filtered_obstacles;
                    continue;
                }

                const auto& obj_pose = obj->shape_poses_[i];
                ObstacleInfo info;
                info.center = obj_pose.translation();
                info.size = size;
                info.orientation = obj_pose.rotation();
                info.shape = std::string(shape_type) == "sphere" ? ObstacleShape::kSphere :
                    (std::string(shape_type) == "cylinder" ? ObstacleShape::kCylinder : ObstacleShape::kBox);
                obstacles.push_back(info);
                object_added = true;

                RCLCPP_DEBUG(logger_,
                    "  obstacle '%s' shape[%zu] type=%s center=[%.4f,%.4f,%.4f] size=[%.4f,%.4f,%.4f]",
                    obj_id.c_str(), i, shape_type,
                    info.center.x(), info.center.y(), info.center.z(),
                    info.size.x(), info.size.y(), info.size.z());
            }
            if (!object_added && shape_count == 0U) {
                ++filtered_obstacles;
            }
        }
    }

    PlanRequestCore plan_req;
    plan_req.q_start = q_start;
    plan_req.q_goal = q_goal;
    plan_req.goal_candidates = goal_candidates;
    plan_req.p_start = p_start;
    plan_req.p_goal = p_goal;
    plan_req.R_target = R_target;
    plan_req.obs_origin = obs_origin;
    plan_req.obs_size = obs_size;
    plan_req.obstacles = obstacles;
    plan_req.use_multi_obstacle = !obstacles.empty();
    plan_req.tool_model = tool_model;
    plan_req.random_seed = options.planner_random_seed == 0U ? 7U : options.planner_random_seed;
    plan_req.started = request_started;
    plan_req.deadline = request_deadline;
    plan_req.cancel_requested = options.cancel_requested;
    plan_req.anytime_checkpoints_s = options.anytime_checkpoints_s;

    if (plan_req.use_multi_obstacle) {
        plan_req.obs_origin = obstacles.front().center;
        plan_req.obs_size = obstacles.front().size;
    }

    RCLCPP_INFO(
        logger_,
        "Planning obstacles aggregated: obs_count=%zu filtered=%zu multi_obs_enabled=%s",
        plan_req.obstacles.size(),
        filtered_obstacles,
        plan_req.use_multi_obstacle ? "true" : "false");

    const std::string planner_name = algorithm->name();
    if (request_should_stop()) {
        RCLCPP_WARN(logger_, "Planning deadline or cancellation reached before core search");
        res.planning_time_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - request_started).count();
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::TIMED_OUT;
        return false;
    }
    RCLCPP_INFO(
        logger_,
        "Planner branch selected: %s %s",
        planner_name.c_str(),
        plan_req.use_multi_obstacle ? "multi" : "single");

    auto result = engine.plan(plan_req);
    if (request_should_stop() && result.success) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kTimeout;
        result.message = "deadline_reached_after_core_search";
    }
    if (result.effort_stats.work_units <= 0) {
        result.effort_stats.work_units = std::max(0, result.iterations);
    }
    result.effort_stats.goal_root_count = static_cast<int>(goal_candidates.size());
    int selected_planning_root = -1;
    if (result.success && !result.path.empty()) {
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (std::size_t root_index = 0U; root_index < goal_candidates.size(); ++root_index) {
            const double distance = (result.path.back() - goal_candidates[root_index]).norm();
            if (distance < nearest_distance) {
                nearest_distance = distance;
                selected_planning_root = static_cast<int>(root_index);
            }
        }
    }
    result.collision_state_checks = planning_collision->stateChecks();
    result.collision_motion_checks = planning_collision->motionChecks();
    result.valid_motion_edges = planning_collision->validMotionEdges();
    result.invalid_motion_edges = planning_collision->invalidMotionEdges();
    result.collision_clearance_checks = planning_collision->clearanceChecks();
    appendSamplingStats(result, plan_req.random_seed);
    appendAnytimeTrace(result, plan_req.random_seed);
    // MIRE diagnostics use the planner's accepted-root index. The public
    // benchmark result uses the index in root_diagnostics.csv so that the two
    // artifacts can be joined even when rejected IK candidates precede it.
    result.effort_stats.selected_goal_root = selected_planning_root;
    appendMireStats(result, plan_req.random_seed);
    result.effort_stats.selected_goal_root = -1;
    if (selected_planning_root >= 0) {
        const auto& selected_root = goal_candidates[static_cast<std::size_t>(selected_planning_root)];
        if (goal_root_diagnostics.empty()) {
            result.effort_stats.selected_goal_root = selected_planning_root;
        } else {
            for (std::size_t diagnostic_index = 0U;
                 diagnostic_index < goal_root_diagnostics.size(); ++diagnostic_index) {
                if ((goal_root_diagnostics[diagnostic_index].q - selected_root).norm() < 1e-8) {
                    result.effort_stats.selected_goal_root = static_cast<int>(diagnostic_index);
                    break;
                }
            }
        }
    }
    appendRootDiagnostics(goal_candidates, goal_root_diagnostics, q_start, result, plan_req.random_seed);
    if (!result.diagnostics.empty()) {
        RCLCPP_INFO(logger_, "%s", result.diagnostics.c_str());
    }

    if (!result.success) {
        RCLCPP_WARN(
            logger_,
            "Fairino plan failure: planner=%s planning_time=%.6f path_points=%zu num_nodes=%d iterations=%d message=%s",
            planner_name.c_str(),
            result.planning_time,
            result.path.size(),
            result.num_nodes,
            result.iterations,
            result.message.c_str());
        RCLCPP_INFO(logger_, "PathOptimizer: skipped (planning failed)");
        RCLCPP_INFO(logger_, "TrajectorySmoother: skipped (planning failed before trajectory export)");
        res.planning_time_ = result.planning_time;
        appendPlannerDiagnostics(result, plan_req.random_seed, planner_name, false, 0.0, 0.0, 0.0);
        res.error_code_.val = toMoveItError(result.failure_code);
        return false;
    }

    auto raw_path = result.path;
    if (!raw_path.empty()) {
        const double start_alignment_max_error =
            (raw_path.front() - q_scene_current).cwiseAbs().maxCoeff();
        if (start_alignment_max_error > 1e-6 &&
            start_alignment_max_error <= 0.05 &&
            collision->isStateValid(q_scene_current) &&
            (raw_path.size() < 2U ||
             collision->isMotionValid(
                 q_scene_current, raw_path[1U], final_validation_distance))) {
            RCLCPP_INFO(
                logger_,
                "StartStateAlign: replacing path start with PlanningScene current state (max_error_rad=%.5f)",
                start_alignment_max_error);
            raw_path.front() = q_scene_current;
        }
    }

    auto path = raw_path;
    const size_t raw_path_points = path.size();
    const double raw_path_cost = std::isfinite(result.path_cost)
        ? result.path_cost
        : jointPathLength(path);
    RCLCPP_INFO(
        logger_,
        "Fairino plan result: planner=%s planning_time=%.6f path_points=%zu path_cost=%.6f num_nodes=%d iterations=%d",
        planner_name.c_str(),
        result.planning_time,
        path.size(),
        result.path_cost,
        result.num_nodes,
        result.iterations);
    double path_optimizer_time_s = 0.0;
    if (path.size() > 2 && options.enable_path_optimizer) {
        OrientationPolicy ori_policy;
        OrientationChecker ori_checker(ori_policy);
        ori_checker.setTargetOrientation(R_target);
        ori_checker.setTargetPosition(p_goal);
        ori_checker.setToolModel(tool_model);
        PathOptimizer optimizer;
        optimizer.setCollisionChecker(collision.get());
        optimizer.setOrientationChecker(ori_checker);
        optimizer.setToolTransform(flange_to_tool);
        optimizer.setJointLimits(JointLimits{});
        optimizer.setValidationDistance(std::min(
            options.optimizer_validation_distance, final_validation_distance));
        optimizer.setFailOpenReturnOriginal(options.optimizer_fail_open_return_original);
        optimizer.setDensifyMaxSpacing(options.optimizer_densify_max_spacing);
        optimizer.setPullAlphaRange(
            options.optimizer_pull_alpha_min,
            options.optimizer_pull_alpha_max);
        optimizer.setOrientationCheckCount(options.optimizer_orientation_check_count);
        const size_t optimizer_input_points = path.size();
        const auto optimizer_start = std::chrono::steady_clock::now();
        path = optimizer.optimize(
            path,
            options.optimizer_shortcut_trials,
            options.optimizer_pull_trials);
        path_optimizer_time_s =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - optimizer_start).count();
        RCLCPP_INFO(
            logger_,
            "PathOptimizer: input_points=%zu output_points=%zu time_s=%.6f",
            optimizer_input_points,
            path.size(),
            path_optimizer_time_s);
    } else if (path.size() > 2) {
        RCLCPP_INFO(logger_, "PathOptimizer disabled by planner.enable_path_optimizer=false");
    }

    // Only densify very sparse export paths.  Broad post-validation densify can
    // introduce new invalid segments on already validated multi-waypoint paths.
    if (path.size() <= 3U) {
        const size_t path_before_execution_densify = path.size();
        path = densifyPathForExecution(
            path, std::max(1e-4, options.optimizer_densify_max_spacing));
        if (path.size() != path_before_execution_densify) {
            RCLCPP_INFO(
                logger_,
                "TrajectoryExecutionDensify: input_points=%zu output_points=%zu max_spacing=%.4f",
                path_before_execution_densify,
                path.size(),
                std::max(1e-4, options.optimizer_densify_max_spacing));
        }
    }

    const auto final_validation_started = std::chrono::steady_clock::now();
    int invalid_segment = -1;
    bool final_path_valid = validateJointPath(
        path, collision, final_validation_distance, &invalid_segment);
    if (!final_path_valid && !options.final_validation_fail_open) {
        int raw_invalid_segment = -1;
        const bool raw_path_valid = validateJointPath(
            raw_path, collision, final_validation_distance, &raw_invalid_segment);
        if (raw_path_valid) {
            RCLCPP_WARN(
                logger_,
                "FinalPathValidator rejected optimized path; using raw planner path instead");
            path = raw_path;
            final_path_valid = true;
            invalid_segment = -1;
        } else {
            RCLCPP_WARN(
                logger_,
                "FinalPathValidator raw-path fallback unavailable: raw_invalid_segment=%d",
                raw_invalid_segment);
        }
    }
    const double final_validation_time_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - final_validation_started).count();
    RCLCPP_INFO(
        logger_,
        "FinalPathValidator: points=%zu valid=%s invalid_segment=%d validation_distance=%.4f fail_open=%s",
        path.size(),
        final_path_valid ? "true" : "false",
        invalid_segment,
        final_validation_distance,
        options.final_validation_fail_open ? "true" : "false");
    RCLCPP_INFO(
        logger_,
        "PathQuality: planner=%s raw_points=%zu raw_cost=%.6f optimized_points=%zu optimized_length=%.6f final_valid=%s",
        planner_name.c_str(),
        raw_path_points,
        raw_path_cost,
        path.size(),
        jointPathLength(path),
        final_path_valid ? "true" : "false");
    if (!final_path_valid && !options.final_validation_fail_open) {
        RCLCPP_ERROR(
            logger_,
            "FinalPathValidator rejected path; refusing trajectory export and execution");
        RCLCPP_INFO(logger_, "TrajectorySmoother: skipped (final path validation failed)");
        res.planning_time_ = result.planning_time;
        appendPlannerDiagnostics(result, plan_req.random_seed, planner_name, false, raw_path_cost, 0.0, 0.0);
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        return false;
    }

    const auto decimator_start = std::chrono::steady_clock::now();
    const size_t decimator_input_points = path.size();
    auto export_path = decimatePathForExport(path, collision, final_validation_distance);
    int decimator_invalid_segment = -1;
    const bool decimator_valid = validateJointPath(
        export_path, collision, final_validation_distance, &decimator_invalid_segment);
    const double decimator_time_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decimator_start).count();
    const double export_max_segment = std::max(0.02, final_validation_distance * 2.0);
    RCLCPP_INFO(
        logger_,
        "TrajectoryExportDecimator: input_points=%zu output_points=%zu validated=%s length=%.6f invalid_segment=%d max_segment=%.4f time_s=%.6f",
        decimator_input_points,
        export_path.size(),
        decimator_valid ? "true" : "false",
        jointPathLength(export_path),
        decimator_invalid_segment,
        export_max_segment,
        decimator_time_s);
    if (!decimator_valid && !options.final_validation_fail_open) {
        RCLCPP_ERROR(
            logger_,
            "TrajectoryExportDecimator produced invalid export path; refusing trajectory export");
        res.planning_time_ = result.planning_time;
        appendPlannerDiagnostics(result, plan_req.random_seed, planner_name, false, raw_path_cost, 0.0, 0.0);
        res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
        return false;
    }
    path = std::move(export_path);

    const double nominal_waypoint_dt = std::max(1e-4, options.trajectory_waypoint_dt);
    const double velocity_scaling = std::clamp(
        req.max_velocity_scaling_factor > 0.0 ? req.max_velocity_scaling_factor : 1.0,
        1e-3, 1.0);
    const double acceleration_scaling = std::clamp(
        req.max_acceleration_scaling_factor > 0.0 ? req.max_acceleration_scaling_factor : 1.0,
        1e-3, 1.0);
    const auto build_trajectory = [&](
        const std::vector<JointConfig>& candidate_path,
        bool* retimed_out) {
        auto candidate = std::make_shared<robot_trajectory::RobotTrajectory>(
            scene->getRobotModel(), group_name);
        for (size_t i = 0; i < candidate_path.size(); ++i) {
            moveit::core::RobotState state(scene->getRobotModel());
            state = scene->getCurrentState();
            std::vector<double> vals(
                candidate_path[i].data(), candidate_path[i].data() + NUM_JOINTS);
            state.setJointGroupPositions(jmg, vals);
            state.update();
            const double dt = (i == 0) ? 0.0 : nominal_waypoint_dt;
            candidate->addSuffixWayPoint(state, dt);
        }
        trajectory_processing::TimeOptimalTrajectoryGeneration totg;
        const bool retimed_candidate = totg.computeTimeStamps(
            *candidate, velocity_scaling, acceleration_scaling);
        if (retimed_out) {
            *retimed_out = retimed_candidate;
        }
        return candidate;
    };

    const auto export_start = std::chrono::steady_clock::now();
    bool retimed = false;
    auto traj = build_trajectory(path, &retimed);
    const double export_time_s =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - export_start).count();
    RCLCPP_INFO(
        logger_,
        "TrajectoryExport: points=%zu time_s=%.6f retimed=%s velocity_scaling=%.3f acceleration_scaling=%.3f",
        path.size(), export_time_s, retimed ? "true" : "false",
        velocity_scaling, acceleration_scaling);

    // Validate the exact RobotTrajectory returned to MoveIt, not only the
    // JointConfig representation used by the planner and optimizer.
    std::vector<std::size_t> moveit_invalid_indices;
    if (!scene->isPathValid(*traj, group_name, false, &moveit_invalid_indices)) {
        const auto invalid_index = moveit_invalid_indices.empty()
            ? -1LL
            : static_cast<long long>(moveit_invalid_indices.front());
        bool raw_retimed = false;
        auto raw_traj = build_trajectory(raw_path, &raw_retimed);
        std::vector<std::size_t> raw_invalid_indices;
        if (scene->isPathValid(*raw_traj, group_name, false, &raw_invalid_indices)) {
            RCLCPP_WARN(
                logger_,
                "MoveItTrajectoryValidator rejected exported path: points=%zu invalid_index=%lld; using raw planner path points=%zu",
                path.size(), invalid_index, raw_path.size());
            path = raw_path;
            traj = raw_traj;
            retimed = raw_retimed;
        } else {
            const auto raw_invalid_index = raw_invalid_indices.empty()
                ? -1LL
                : static_cast<long long>(raw_invalid_indices.front());
            RCLCPP_ERROR(
                logger_,
                "MoveItTrajectoryValidator rejected exported path: points=%zu invalid_index=%lld raw_points=%zu raw_invalid_index=%lld",
                path.size(), invalid_index, raw_path.size(), raw_invalid_index);
            res.planning_time_ = result.planning_time;
            appendPlannerDiagnostics(result, plan_req.random_seed, planner_name, false, raw_path_cost, 0.0, 0.0);
            res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
            return false;
        }
    }
    if (!retimed) {
        RCLCPP_WARN(
            logger_,
            "Trajectory retiming failed; executing nominal waypoint timing instead");
    }
    RCLCPP_INFO(
        logger_,
        "MoveItTrajectoryValidator: points=%zu valid=true",
        path.size());

    appendTrajectoryPaths(raw_path, path, flange_to_tool, tool_model, plan_req.random_seed);
    res.trajectory_ = traj;
    res.planning_time_ = result.planning_time;
    appendPlannerDiagnostics(
        result, plan_req.random_seed, planner_name, true, raw_path_cost,
        jointPathLength(path), tcpPathLength(path, flange_to_tool, tool_model),
        path_optimizer_time_s,
        jointTurnVariation(path), static_cast<int>(path.size()),
        ik_time_s, root_generation_time_s, final_validation_time_s, export_time_s);
    RCLCPP_INFO(
        logger_,
        "TrajectoryTiming: Fairino global path uses TOTG scaling velocity=%.3f acceleration=%.3f",
        velocity_scaling, acceleration_scaling);
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
}

}  // namespace fairino_planning::v2
