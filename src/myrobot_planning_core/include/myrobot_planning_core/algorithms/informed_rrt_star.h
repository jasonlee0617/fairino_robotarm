#pragma once

#include <random>

#include "myrobot_planning_core/algorithms/planning_algorithm.h"

namespace fairino_planning {

class InformedRRTStar : public PlanningAlgorithm {
public:
    InformedRRTStar();

    PlanResult plan(const PlanRequestCore& request) override;

    PlanResult plan(
        const JointConfig& q_start,
        const JointConfig& q_goal,
        const Vector3d& p_start,
        const Vector3d& p_goal,
        const RotMatrix3d& R_target,
        const Vector3d& obs_origin,
        const Vector3d& obs_size) override;

    std::string name() const override { return "informed_rrt*"; }

private:
    PlanResult planOnce(
        const JointConfig& q_start,
        const std::vector<JointConfig>& goal_candidates,
        unsigned int request_seed,
        const PlanRequestCore* request = nullptr);

    std::mt19937 rng_;
};

}  // namespace fairino_planning
