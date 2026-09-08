#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "myrobot_planning_core/config/defaults.hpp"
#include "myrobot_planning_core/types/aliases.hpp"

namespace fairino_planning {

struct JointLimits {
    JointConfig lower;
    JointConfig upper;

    JointLimits() {
        lower << -3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543;
        upper << 3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543;
    }

    JointLimits(const JointConfig& lower_in, const JointConfig& upper_in)
        : lower(lower_in), upper(upper_in) {}

    bool isWithin(const JointConfig& q, double tol = 1e-6) const {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (q[i] < lower[i] - tol || q[i] > upper[i] + tol) {
                return false;
            }
        }
        return true;
    }

    JointConfig clamp(const JointConfig& q) const { return q.cwiseMax(lower).cwiseMin(upper); }

    JointConfig sampleUniform(std::mt19937& rng) const {
        JointConfig q;
        for (int i = 0; i < NUM_JOINTS; ++i) {
            std::uniform_real_distribution<double> dist(lower[i], upper[i]);
            q[i] = dist(rng);
        }
        return q;
    }
};

enum class ObstacleShape { kBox, kSphere, kCylinder };

struct ObstacleInfo {
    Vector3d center{Vector3d::Zero()};
    Vector3d size{Vector3d::Zero()};
    ObstacleShape shape{ObstacleShape::kBox};
    RotMatrix3d orientation{RotMatrix3d::Identity()};
};

struct AapfParams {
    bool enable = true;
    double repulsion_range_m = 0.24;
    int stall_threshold_iters = 120;
    double strict_validation_distance = 0.03;
    int adaptive_window = 32;
    double adaptive_exploration = 0.7;
    int global_min_period = 8;
    double min_rewire_radius_ratio = 1.2;
    double shrink_motion_initial_scale = 0.5;
    double shrink_motion_decay = 0.5;
    int shrink_motion_attempts = 4;
    double bridge_node_sep_ratio = 0.25;
};

// Parameters limited to MIRE's batch graph and effort-focal search.
struct MireBiAitParams {
    int batch_size = 128;
    int post_solution_batch_size = 10;
    double rgg_factor = 1.1;
    double effort_focal_factor = 1.10;
    std::string ablation_variant = "full";
    bool enable_effort_focal_queue = true;
    bool enable_lazy_edge_validation = true;
};

struct PrmParams {
    int k_neighbors = 12;
};

struct PlanningParams {
    // Non-binding protection against a defective/non-cooperative search. The
    // formal benchmark is terminated by PlanRequestCore::deadline.
    int max_iterations = 50000;
    // Zero preserves ordinary-planning behavior. Benchmark overrides this to a
    // fixed count of sampling attempts after the first strict solution.
    int post_solution_sample_attempts = 0;
    double max_step = 0.20;
    double goal_threshold = 0.08;
    double goal_bias = 0.20;
    int max_ik_tries = 2;
    double gamma = 1.5;
    double max_rewire_radius = 0.25;
    int max_near = 20;

    int connect_max_steps = 15;
    double connect_goal_bias = 0.12;
    int rewire_every_k = 3;
    int rewire_max_neighbors = 5;

    double validation_distance = 0.05;
    int connect_success_every_k = 3;
    double connect_success_dist_scale = 1.5;
    double direct_connect_step_factor = 1.5;
    double connect_target_tolerance = 0.02;

    AapfParams aapf;
    MireBiAitParams mire_biait;
    PrmParams prm;
};

struct OrientationFallbackLevel {
    double ori_near_tol_deg = defaults::kOriNearTolDeg;
    double near_dist = defaults::kNearDist;
    double ori_gate_dist = defaults::kOriGateDist;
};

struct OrientationPolicy {
    double near_dist = defaults::kNearDist;
    double ori_gate_dist = defaults::kOriGateDist;
    double ori_far_tol_deg = defaults::kOriFarTolDeg;
    double ori_near_tol_deg = defaults::kOriNearTolDeg;
    double ori_weight_far = defaults::kOriWeightFar;
    double ori_weight_near = defaults::kOriWeightNear;
    std::vector<OrientationFallbackLevel> fallback_levels{
        {1.0, 0.12, 0.12},
        {3.0, 0.15, 0.15},
        {5.0, 0.20, 0.20},
        {10.0, 0.25, 0.25}
    };
};

struct PlannerConfig {
    PlanningParams planning;
    OrientationPolicy orientation;
    JointLimits limits;
};

}  // namespace fairino_planning
