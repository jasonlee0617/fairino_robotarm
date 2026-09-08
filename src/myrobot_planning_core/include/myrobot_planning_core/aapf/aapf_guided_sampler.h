#pragma once

#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/config/planning_params.hpp"
#include "myrobot_planning_core/tree/rrt_tree.h"

#include <array>
#include <optional>
#include <random>
#include <vector>

namespace fairino_planning {

enum class AapfSampleSource {
    kGuided,
    kGlobal,
};

struct AapfJointFieldSample {
    JointConfig goal_dir{JointConfig::Zero()};
    JointConfig away_dir{JointConfig::Zero()};
    JointConfig primary_dir{JointConfig::Zero()};
    JointConfig tangent_dir{JointConfig::Zero()};
    double risk{0.0};
    double step{0.0};
};

AapfJointFieldSample evaluateAapfJointField(
    const JointConfig& q_near,
    const JointConfig& q_target,
    const ClearanceQueryResult& clearance,
    double repulsion_range,
    double max_step);

std::vector<JointConfig> aapfJointCandidateDirections(
    const AapfJointFieldSample& field,
    bool include_tangents);

struct AapfGuidedSample {
    JointConfig q_new{JointConfig::Zero()};
    JointConfig q_near{JointConfig::Zero()};
    int idx_near{-1};
    bool proposal_generated{false};
    bool valid_edge{false};
    bool basic_edge_prevalidated{false};
    int clearance_queries{0};
    double risk{0.0};
};

class AapfGuidedSampler {
public:
    AapfGuidedSampler(
        const PlanningParams& params,
        const JointLimits& limits,
        const CollisionInterface& collision,
        double validation_distance,
        std::mt19937& rng);

    AapfGuidedSample generate(
        const RRTTree& current,
        const RRTTree& opposite,
        const JointConfig& q_target,
        int tree_index,
        int stalled_iterations,
        AapfSampleSource source);

private:
    AapfGuidedSample generateGlobal(
        const RRTTree& current,
        const RRTTree& opposite);
    AapfGuidedSample generateGuided(
        const RRTTree& current,
        const JointConfig& q_target,
        int tree_index,
        int stalled_iterations);
    bool validateWithShrink(
        const JointConfig& from,
        const JointConfig& candidate,
        JointConfig* accepted) const;

    const PlanningParams& params_;
    const JointLimits& limits_;
    const CollisionInterface& collision_;
    double validation_distance_;
    std::mt19937& rng_;
    std::array<std::vector<std::optional<ClearanceQueryResult>>, 2> clearance_cache_;
};

}  // namespace fairino_planning
