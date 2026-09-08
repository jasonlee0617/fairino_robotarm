#include "myrobot_planning_core/algorithms/informed_rrt_star.h"

#include "../common/informed_goal_sampler.hpp"
#include "rrt_star_search.hpp"

namespace fairino_planning {

InformedRRTStar::InformedRRTStar() : rng_(42) {}

PlanResult InformedRRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    return planOnce(
        request.q_start,
        request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
        request.random_seed, &request);
}

PlanResult InformedRRTStar::plan(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const Vector3d& obs_origin,
    const Vector3d& obs_size) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obs_origin;
    (void)obs_size;
    return planOnce(q_start, {q_goal}, 0U);
}

PlanResult InformedRRTStar::planOnce(
    const JointConfig& q_start,
    const std::vector<JointConfig>& goal_candidates,
    unsigned int request_seed,
    const PlanRequestCore* request) {
    if (!collision_) {
        PlanResult result;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "Informed-RRT*: null collision checker.";
        return result;
    }
    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0U ? 42U : request_seed));
    int informed_samples = 0;
    int uniform_samples = 0;
    PlanResult result = runRrtStarSearch(
        q_start, goal_candidates, params_, limits_, *collision_,
        [this, &q_start, &goal_candidates, &informed_samples, &uniform_samples](
            const RrtStarSearchState& state) {
            if (!state.has_solution) {
                ++uniform_samples;
                JointConfig sample = limits_.sampleUniform(rng_);
                if (params_.goal_bias > 0.0) {
                    std::uniform_real_distribution<double> coin(0.0, 1.0);
                    if (coin(rng_) < params_.goal_bias) {
                        std::uniform_int_distribution<size_t> pick(0U, goal_candidates.size() - 1U);
                        sample = goal_candidates[pick(rng_)];
                    }
                }
                return sample;
            }

            ++informed_samples;
            return informed_sampling::multiRootUnion(
                q_start, goal_candidates, state.best_cost, limits_, rng_).value_or(q_start);
        },
        "Informed-RRT*", request);
    result.diagnostics = "informed_samples=" + std::to_string(informed_samples) +
        " uniform_samples=" + std::to_string(uniform_samples);
    return result;
}

}  // namespace fairino_planning
