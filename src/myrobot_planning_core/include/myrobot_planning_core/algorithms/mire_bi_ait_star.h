#pragma once

#include "myrobot_planning_core/algorithms/planning_algorithm.h"

namespace fairino_planning {

// Multi-IK Root-Effort Bidirectional Adaptive Informed Trees.  The planner
// owns a batch graph and never delegates to AAPF or BiRRT*.
class MireBiAitStar final : public PlanningAlgorithm {
public:
    PlanResult plan(const PlanRequestCore& request) override;

    PlanResult plan(
        const JointConfig& q_start, const JointConfig& q_goal,
        const Vector3d& p_start, const Vector3d& p_goal,
        const RotMatrix3d& R_target, const Vector3d& obs_origin,
        const Vector3d& obs_size) override;

    std::string name() const override { return "mire_biait*"; }
};

}  // namespace fairino_planning
