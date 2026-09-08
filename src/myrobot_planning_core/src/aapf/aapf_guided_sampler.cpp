#include "myrobot_planning_core/aapf/aapf_guided_sampler.h"

#include "myrobot_planning_core/algorithms/aapf_birrt_linear_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fairino_planning {
namespace {

constexpr double kDirectionEpsilon = 1e-10;

JointConfig normalizedOrZero(const JointConfig& value) {
    const double norm = value.norm();
    if (!value.allFinite() || norm <= kDirectionEpsilon) {
        return JointConfig::Zero();
    }
    return value / norm;
}

JointConfig orthogonalDirection(const JointConfig& normal) {
    int axis = 0;
    double least_alignment = std::numeric_limits<double>::infinity();
    for (int joint = 0; joint < NUM_JOINTS; ++joint) {
        const double alignment = std::abs(normal[joint]);
        if (alignment < least_alignment) {
            least_alignment = alignment;
            axis = joint;
        }
    }
    JointConfig basis = JointConfig::Zero();
    basis[axis] = 1.0;
    return normalizedOrZero(basis - basis.dot(normal) * normal);
}

}  // namespace

AapfJointFieldSample evaluateAapfJointField(
    const JointConfig& q_near,
    const JointConfig& q_target,
    const ClearanceQueryResult& clearance,
    double repulsion_range,
    double max_step) {
    AapfJointFieldSample out;
    out.goal_dir = normalizedOrZero(
        aapf_birrt_detail::jointDeltaBounded(q_near, q_target));
    out.away_dir = clearance.has_obstacle
        ? normalizedOrZero(clearance.joint_gradient)
        : JointConfig::Zero();

    const double range = std::max(1e-6, repulsion_range);
    if (clearance.has_obstacle && std::isfinite(clearance.signed_distance)) {
        out.risk = std::clamp(1.0 - clearance.signed_distance / range, 0.0, 1.0);
    }
    if (out.away_dir.norm() <= kDirectionEpsilon) {
        out.risk = 0.0;
    }

    out.primary_dir = normalizedOrZero(
        (1.0 - out.risk) * out.goal_dir + out.risk * out.away_dir);
    if (out.primary_dir.norm() <= kDirectionEpsilon) {
        out.primary_dir = out.goal_dir;
    }

    const JointConfig tangent_normal = out.away_dir.norm() > kDirectionEpsilon
        ? out.away_dir : out.goal_dir;
    out.tangent_dir = normalizedOrZero(
        out.goal_dir - out.goal_dir.dot(tangent_normal) * tangent_normal);
    if (out.tangent_dir.norm() <= kDirectionEpsilon &&
        tangent_normal.norm() > kDirectionEpsilon) {
        out.tangent_dir = orthogonalDirection(tangent_normal);
    }

    out.step = std::max(1e-6, max_step) * (0.25 + 0.75 * (1.0 - out.risk));
    return out;
}

std::vector<JointConfig> aapfJointCandidateDirections(
    const AapfJointFieldSample& field,
    bool include_tangents) {
    std::vector<JointConfig> directions;
    if (field.primary_dir.norm() > kDirectionEpsilon) {
        directions.push_back(field.primary_dir);
    }
    if (!include_tangents || field.tangent_dir.norm() <= kDirectionEpsilon) {
        return directions;
    }
    JointConfig away_component = JointConfig::Zero();
    if (field.away_dir.norm() > kDirectionEpsilon) {
        away_component = std::max(0.1, field.risk) * field.away_dir;
    }
    directions.push_back(normalizedOrZero(field.tangent_dir + away_component));
    directions.push_back(normalizedOrZero(-field.tangent_dir + away_component));
    return directions;
}

AapfGuidedSampler::AapfGuidedSampler(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    std::mt19937& rng)
    : params_(params)
    , limits_(limits)
    , collision_(collision)
    , validation_distance_(std::max(1e-6, validation_distance))
    , rng_(rng) {}

AapfGuidedSample AapfGuidedSampler::generate(
    const RRTTree& current,
    const RRTTree& opposite,
    const JointConfig& q_target,
    int tree_index,
    int stalled_iterations,
    AapfSampleSource source) {
    return source == AapfSampleSource::kGlobal
        ? generateGlobal(current, opposite)
        : generateGuided(current, q_target, tree_index, stalled_iterations);
}

AapfGuidedSample AapfGuidedSampler::generateGlobal(
    const RRTTree& current,
    const RRTTree& opposite) {
    AapfGuidedSample out;
    JointConfig sample;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (opposite.size() > 0 &&
        unit(rng_) < std::clamp(params_.connect_goal_bias, 0.0, 1.0)) {
        std::uniform_int_distribution<int> index(0, opposite.size() - 1);
        sample = opposite.node(index(rng_)).state;
    } else {
        sample = limits_.sampleUniform(rng_);
    }
    out.idx_near = aapf_birrt_detail::nearestBoundedLinear(current, sample);
    if (out.idx_near < 0) return out;
    out.q_near = current.node(out.idx_near).state;
    out.q_new = aapf_birrt_detail::steerBoundedLinear(
        out.q_near, sample, params_.max_step, limits_);
    out.proposal_generated = true;
    return out;
}

AapfGuidedSample AapfGuidedSampler::generateGuided(
    const RRTTree& current,
    const JointConfig& q_target,
    int tree_index,
    int stalled_iterations) {
    AapfGuidedSample out;
    if (tree_index < 0 || tree_index >= static_cast<int>(clearance_cache_.size())) {
        return out;
    }
    out.idx_near = aapf_birrt_detail::nearestBoundedLinear(current, q_target);
    if (out.idx_near < 0) return out;
    out.q_near = current.node(out.idx_near).state;

    auto& cache = clearance_cache_[tree_index];
    if (cache.size() < static_cast<std::size_t>(current.size())) {
        cache.resize(static_cast<std::size_t>(current.size()));
    }
    if (!cache[out.idx_near]) {
        cache[out.idx_near] = collision_.nearestWorldClearance(
            out.q_near, params_.aapf.repulsion_range_m);
        out.clearance_queries = 1;
    }
    const ClearanceQueryResult& clearance = *cache[out.idx_near];
    if (!clearance.supported) return out;

    const AapfJointFieldSample field = evaluateAapfJointField(
        out.q_near, q_target, clearance,
        params_.aapf.repulsion_range_m, params_.max_step);
    out.risk = field.risk;

    const bool escape = field.risk > 0.0 ||
        stalled_iterations >= std::max(1, params_.aapf.stall_threshold_iters);
    const std::vector<JointConfig> directions =
        aapfJointCandidateDirections(field, escape);

    for (const JointConfig& direction : directions) {
        if (direction.norm() <= kDirectionEpsilon) continue;
        const JointConfig candidate = limits_.clamp(out.q_near + field.step * direction);
        out.proposal_generated = true;
        JointConfig accepted;
        if (validateWithShrink(out.q_near, candidate, &accepted)) {
            out.q_new = accepted;
            out.valid_edge = true;
            out.basic_edge_prevalidated = true;
            return out;
        }
    }
    return out;
}

bool AapfGuidedSampler::validateWithShrink(
    const JointConfig& from,
    const JointConfig& candidate,
    JointConfig* accepted) const {
    if (!accepted) return false;
    const auto valid = [&](const JointConfig& state) {
        return collision_.isStateValid(state) &&
               collision_.isMotionValid(from, state, validation_distance_);
    };
    if (valid(candidate)) {
        *accepted = candidate;
        return true;
    }

    double scale = params_.aapf.shrink_motion_initial_scale;
    for (int attempt = 0; attempt < params_.aapf.shrink_motion_attempts;
         ++attempt, scale *= params_.aapf.shrink_motion_decay) {
        const JointConfig state = limits_.clamp(from + scale * (candidate - from));
        if ((state - from).norm() <= 1e-4 || !valid(state)) continue;
        *accepted = state;
        return true;
    }
    return false;
}

}  // namespace fairino_planning
