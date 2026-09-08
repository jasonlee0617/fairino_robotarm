#pragma once

#include <limits>
#include <string>
#include <vector>

#include "myrobot_planning_core/types/aliases.hpp"

namespace fairino_planning {

enum class PlanningFailureCode {
    kNone = 0,
    kInvalidInput,
    kGoalNotReached,
    kCollision,
    kIKFailed,
    kTimeout,
    kInternalError
};

struct AnytimeTracePoint {
    double elapsed_s = 0.0;
    bool has_solution = false;
    double incumbent_joint_cost_rad = std::numeric_limits<double>::infinity();
    int iterations = 0;
    int sample_attempts = 0;
    int accepted_samples = 0;
    int num_nodes = 0;
    int work_units = 0;
};

// Per-tree, per-source accounting for planners that adapt a sampling budget.
// Empty for planners that do not expose sampling arms.
struct SamplingStats {
    int tree_index = 0;
    std::string source;
    int selections = 0;
    int proposals = 0;
    int insertions = 0;
    int progress_events = 0;
    int connections = 0;
    int improvements = 0;
    int clearance_queries = 0;
    double utility_sum = 0.0;
    double elapsed_ms = 0.0;
};

// Minimal auditable work counters for batch/lazy planners.  Other planners
// leave this at zero; benchmark ranking remains in results.csv.
struct PlanningEffortStats {
    int batches = 0;
    int work_units = 0;
    int sample_attempts = 0;
    int sampled_states = 0;
    int uniform_sample_attempts = 0;
    int informed_sample_attempts = 0;
    int uniform_accepted_states = 0;
    int informed_accepted_states = 0;
    int graph_edges = 0;
    int reverse_queue_pops = 0;
    int reverse_queue_stale_discards = 0;
    int reverse_queue_consistent_discards = 0;
    int forward_edge_pops = 0;
    int goal_tree_edge_pops = 0;
    int sparse_state_checks = 0;
    int edge_validation_attempts = 0;
    int final_validation_calls = 0;
    int state_validation_calls = 0;
    int blocked_edges = 0;
    int focal_entries_examined = 0;
    int goal_root_count = 0;
    int selected_goal_root = -1;
    int incumbent_improvements = 0;
    int post_solution_sample_attempts = 0;
    int informed_path_improvements = 0;
    int post_cost_search_pops = 0;
    int post_candidate_paths = 0;
    int post_strict_improvements = 0;
    int sparse_rejected_edges = 0;
    int quarter_state_checks = 0;
    int full_invalid_edges_avoided = 0;
    int selected_goal_root_before = -1;
    int selected_goal_root_after = -1;
    bool budget_exhausted = false;
    bool post_solution_budget_complete = false;
    bool lower_bound_certified = false;
    double first_solution_time_s = 0.0;
    double first_solution_path_cost = std::numeric_limits<double>::infinity();
    double joint_cost_reduction_rad = 0.0;
    double tcp_cost_reduction_m = 0.0;
    std::vector<double> root_lower_bounds;
    std::vector<int> root_sample_attempts;
    std::vector<int> root_accepted_samples;
    std::vector<int> root_path_improvements;
};

struct PlanResultCore {
    bool success = false;
    std::vector<JointConfig> path;
    std::vector<JointConfig> trajectory;
    std::vector<double> timestamps;
    double planning_time = 0.0;
    double path_cost = std::numeric_limits<double>::infinity();
    int iterations = 0;
    int num_nodes = 0;
    int sample_attempts = 0;
    int accepted_samples = 0;
    int collision_state_checks = 0;
    int collision_motion_checks = 0;
    int valid_motion_edges = 0;
    int invalid_motion_edges = 0;
    int collision_clearance_checks = 0;
    bool lower_bound_certified = false;
    double first_solution_time_s = 0.0;
    double first_solution_path_cost = std::numeric_limits<double>::infinity();
    std::string message;
    std::string stop_reason;
    std::string diagnostics;
    std::vector<AnytimeTracePoint> anytime_trace;
    std::vector<SamplingStats> sampling_stats;
    PlanningEffortStats effort_stats;
    PlanningFailureCode failure_code = PlanningFailureCode::kNone;
};

using PlanResult = PlanResultCore;

}  // namespace fairino_planning
