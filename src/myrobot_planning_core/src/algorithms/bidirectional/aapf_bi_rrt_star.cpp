#include "myrobot_planning_core/algorithms/aapf_bi_rrt_star.h"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"

#include "../common/goal_set_utils.hpp"
#include "../common/bidirectional_utils.hpp"
#include "myrobot_planning_core/algorithms/aapf_birrt_linear_ops.hpp"
#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/model/robot_kinematics_config.hpp"
#include "myrobot_planning_core/aapf/aapf_adaptive_sampler_selector.h"
#include "myrobot_planning_core/aapf/aapf_guided_sampler.h"
#include "myrobot_planning_core/tree/rrt_tree.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace fairino_planning {

using aapf_birrt_detail::jointDeltaBounded;
using aapf_birrt_detail::jointDistance;
using aapf_birrt_detail::jointDistanceSq;
using aapf_birrt_detail::kEpsDistZero;
using aapf_birrt_detail::nearestBoundedLinear;
using aapf_birrt_detail::steerBoundedLinear;

// ── Numerical stability constants (not exposed as YAML) ──
constexpr double kEpsJointNear = 1e-4;
constexpr double kEpsBridgeSep = 1e-5;
constexpr double kEpsDuplicatePathPoint = 1e-10;
constexpr double kEpsJointLimitTol = 1e-4;
constexpr double kEpsCostEqual = 1e-12;

namespace {
struct ConnectionQuality {
    bool valid{false};
    bool improved{false};
};

std::vector<int> nearRadiusBoundedLinear(
    const RRTTree& tree,
    const JointConfig& q,
    double radius) {
    std::vector<int> result;
    const double r2 = radius * radius;
    for (int i = 0; i < tree.size(); ++i) {
        if (jointDistanceSq(tree.node(i).state, q) <= r2) {
            result.push_back(i);
        }
    }
    return result;
}

double minimumJointStep(const PlanningParams& params) {
    return std::max(kEpsJointNear, 0.05 * std::max(kEpsJointNear, params.max_step));
}

int secondNearestBoundedLinear(const RRTTree& tree, const JointConfig& q, int skip_idx) {
    int best = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < tree.size(); ++i) {
        if (i == skip_idx) continue;
        const double d2 = jointDistanceSq(tree.node(i).state, q);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

struct ConnectionResult {
    bool connected = false;
    bool advanced = false;
    JointConfig q_last_valid{JointConfig::Zero()};
    std::vector<JointConfig> bridge;
    double edge_dist = 0.0;
    double advanced_dist = 0.0;
    int idx_other = -1;
};

struct PathValidator {
    const CollisionInterface& collision;
    const double basic_distance;
    const double strict_distance;

    static bool finite(const JointConfig& q) {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            if (!std::isfinite(q[i])) return false;
        }
        return true;
    }

    bool basic(const JointConfig& from, const JointConfig& to) const {
        return finite(from) && finite(to) && collision.isStateValid(from) &&
               collision.isStateValid(to) && collision.isMotionValid(from, to, basic_distance);
    }

    bool strict(const JointConfig& from, const JointConfig& to) const {
        return finite(from) && finite(to) && collision.isStateValid(from) &&
               collision.isStateValid(to) && collision.isMotionValid(from, to, strict_distance);
    }

    bool strictPath(const std::vector<JointConfig>& path, int* bad_segment = nullptr) const {
        if (bad_segment) *bad_segment = -1;
        if (path.empty()) return false;
        if (!finite(path.front()) || !collision.isStateValid(path.front())) {
            if (bad_segment) *bad_segment = 0;
            return false;
        }
        for (size_t i = 1; i < path.size(); ++i) {
            if (!strict(path[i - 1U], path[i])) {
                if (bad_segment) *bad_segment = static_cast<int>(i - 1U);
                return false;
            }
        }
        return true;
    }

    static double cost(const std::vector<JointConfig>& path) {
        double total = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            total += jointDistance(path[i - 1U], path[i]);
        }
        return total;
    }

    bool finalize(std::vector<JointConfig>* path) const {
        return path && !path->empty() && strictPath(*path);
    }
};

double computeRewireRadius(const PlanningParams& params, int n) {
    return bidirectional::rewireRadius(params, n, params.aapf.min_rewire_radius_ratio);
}

int extendRrtStar(
    RRTTree& tree,
    const JointConfig& q_new,
    int idx_near,
    const PlanningParams& params,
    const CollisionInterface& collision,
    double validation_distance,
    bool allow_near_fallback,
    bool rewire) {
    auto near_set = nearRadiusBoundedLinear(tree, q_new, computeRewireRadius(params, tree.size()));
    if (near_set.empty()) near_set.push_back(idx_near);
    if (static_cast<int>(near_set.size()) > params.max_near) {
        std::partial_sort(
            near_set.begin(), near_set.begin() + params.max_near, near_set.end(),
            [&](int a, int b) {
                return jointDistanceSq(tree.node(a).state, q_new) <
                       jointDistanceSq(tree.node(b).state, q_new);
            });
        near_set.resize(params.max_near);
    }

    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(near_set.size());
    for (int idx : near_set) {
        candidates.emplace_back(
            tree.node(idx).cost + jointDistance(tree.node(idx).state, q_new), idx);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    int parent = -1;
    double parent_cost = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        if (collision.isMotionValid(tree.node(candidate.second).state, q_new, validation_distance)) {
            parent = candidate.second;
            parent_cost = candidate.first;
            break;
        }
    }
    if (parent < 0) {
        if (!allow_near_fallback) return -1;
        parent = idx_near;
        parent_cost = tree.node(parent).cost + jointDistance(tree.node(parent).state, q_new);
    }

    const int new_idx = tree.addNode(q_new, parent, parent_cost);
    if (!rewire) return new_idx;
    const int rewire_count = std::min(params.rewire_max_neighbors,
                                      static_cast<int>(near_set.size()));
    for (int i = 0; i < rewire_count; ++i) {
        const int idx = near_set[i];
        if (idx == parent || idx == new_idx) continue;
        const double candidate_cost = tree.node(new_idx).cost +
            jointDistance(q_new, tree.node(idx).state);
        if (candidate_cost + kEpsCostEqual >= tree.node(idx).cost ||
            !collision.isMotionValid(q_new, tree.node(idx).state, validation_distance)) {
            continue;
        }
        tree.reparent(idx, new_idx, candidate_cost);
    }
    return new_idx;
}

bool shrinkMotionToward(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_from,
    const JointConfig& q_to,
    JointConfig* q_out,
    double* dist_out) {
    if (!q_out || !dist_out) return false;
    const JointConfig delta = jointDeltaBounded(q_from, q_to);
    const double min_joint_step = minimumJointStep(params);
    double scale = params.aapf.shrink_motion_initial_scale;
    for (int i = 0; i < params.aapf.shrink_motion_attempts;
         ++i, scale *= params.aapf.shrink_motion_decay) {
        const JointConfig q_try = limits.clamp(q_from + scale * delta);
        const double distance = jointDistance(q_from, q_try);
        if (distance < min_joint_step || !collision.isStateValid(q_try) ||
            !collision.isMotionValid(q_from, q_try, validation_distance)) {
            continue;
        }
        *q_out = q_try;
        *dist_out = distance;
        return true;
    }
    return false;
}

ConnectionResult tryConnectToIndex(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_new,
    RRTTree& other_tree,
    int idx_target) {
    ConnectionResult result;
    if (idx_target < 0) return result;
    result.idx_other = idx_target;
    result.q_last_valid = q_new;
    const JointConfig q_target = other_tree.node(idx_target).state;
    const double distance = jointDistance(q_new, q_target);
    const auto record_advance = [&](const JointConfig& q_current) {
        const double progressed = jointDistance(q_new, q_current);
        if (progressed <= minimumJointStep(params)) return;
        result.advanced = true;
        result.q_last_valid = q_current;
        result.advanced_dist = progressed;
        if (result.bridge.empty() || jointDistance(result.bridge.back(), q_current) >
            std::max(
                kEpsBridgeSep,
                minimumJointStep(params) * params.aapf.bridge_node_sep_ratio)) {
            result.bridge.push_back(q_current);
        }
    };

    if (distance < params.max_step * params.direct_connect_step_factor) {
        if (collision.isMotionValid(q_new, q_target, validation_distance)) {
            result.connected = true;
            result.edge_dist = distance;
            result.q_last_valid = q_target;
        } else {
            JointConfig q_shrunk;
            double shrink_distance = 0.0;
            if (shrinkMotionToward(params, limits, collision, validation_distance, q_new, q_target,
                                   &q_shrunk, &shrink_distance)) {
                record_advance(q_shrunk);
            }
        }
    } else if (distance < params.max_step * params.connect_max_steps) {
        JointConfig q_current = q_new;
        for (int step = 0; step < params.connect_max_steps; ++step) {
            const JointConfig q_step = steerBoundedLinear(q_current, q_target,
                                                          params.max_step, limits);
            if (!collision.isStateValid(q_step) ||
                !collision.isMotionValid(q_current, q_step, validation_distance)) {
                JointConfig q_shrunk;
                double shrink_distance = 0.0;
                if (!shrinkMotionToward(
                        params, limits, collision, validation_distance, q_current, q_target,
                                        &q_shrunk, &shrink_distance)) {
                    break;
                }
                q_current = q_shrunk;
                record_advance(q_current);
                continue;
            }
            q_current = q_step;
            record_advance(q_current);
            if (jointDistance(q_step, q_target) < params.connect_target_tolerance &&
                collision.isMotionValid(q_step, q_target, validation_distance)) {
                result.connected = true;
                result.edge_dist = distance;
                result.q_last_valid = q_step;
                break;
            }
        }
        if (!result.connected &&
            jointDistance(q_current, q_target) <
                params.max_step * params.direct_connect_step_factor &&
            collision.isMotionValid(q_current, q_target, validation_distance)) {
            result.connected = true;
            result.edge_dist = distance;
            result.q_last_valid = q_current;
        }
    }
    return result;
}

ConnectionResult tryConnect(
    const PlanningParams& params,
    const JointLimits& limits,
    const CollisionInterface& collision,
    double validation_distance,
    const JointConfig& q_new,
    RRTTree& other_tree) {
    const int idx_other = nearestBoundedLinear(other_tree, q_new);
    if (idx_other < 0) return {};
    ConnectionResult result = tryConnectToIndex(
        params, limits, collision, validation_distance, q_new, other_tree, idx_other);
    if (result.connected || result.advanced) {
        return result;
    }
    const int idx_second = secondNearestBoundedLinear(other_tree, q_new, idx_other);
    if (idx_second < 0) return result;
    return tryConnectToIndex(
        params, limits, collision, validation_distance, q_new, other_tree, idx_second);
}

int appendConnectionBridge(
    RRTTree& tree,
    int parent,
    const ConnectionResult& connection,
    const PathValidator& validator,
    const PlanningParams& params,
    bool* inserted = nullptr) {
    if (inserted) *inserted = false;
    if (parent < 0) return -1;
    const auto append = [&](const JointConfig& q_bridge) {
        if (jointDistance(tree.node(parent).state, q_bridge) <
            minimumJointStep(params)) {
            return true;
        }
        if (!validator.basic(tree.node(parent).state, q_bridge)) return false;
        parent = tree.addNode(q_bridge, parent,
            tree.node(parent).cost + jointDistance(tree.node(parent).state, q_bridge));
        if (inserted) *inserted = true;
        return true;
    };
    for (const auto& q_bridge : connection.bridge) {
        if (!append(q_bridge)) return -1;
    }
    if (connection.bridge.empty() && connection.advanced && !append(connection.q_last_valid)) {
        return -1;
    }
    return parent;
}

bool buildConnectedPath(
    const RRTTree& tree_a,
    int conn_a,
    const RRTTree& tree_b,
    int conn_b,
    const PathValidator& validator,
    std::vector<JointConfig>* path,
    int* bad_segment = nullptr) {
    if (!path || conn_a < 0 || conn_b < 0) return false;
    auto path_a = tree_a.backtrack(conn_a);
    auto path_b = tree_b.backtrack(conn_b);
    std::reverse(path_b.begin(), path_b.end());
    path->clear();
    path->insert(path->end(), path_a.begin(), path_a.end());
    path->insert(path->end(), path_b.begin(), path_b.end());
    path->erase(std::unique(path->begin(), path->end(),
        [](const JointConfig& a, const JointConfig& b) {
            return (a - b).norm() < kEpsDuplicatePathPoint;
        }), path->end());
    if (validator.finalize(path)) return true;
    validator.strictPath(*path, bad_segment);
    return false;
}

}  // namespace

AapfBiRRTStar::AapfBiRRTStar() : rng_(17) {}

PlanResult AapfBiRRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    return planImpl(
        request.q_start, request.q_goal,
        request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
        request.random_seed, &request);
}

PlanResult AapfBiRRTStar::plan(
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
    return planImpl(q_start, q_goal, {q_goal}, 0);
}

PlanResult AapfBiRRTStar::planMultiObs(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const Vector3d& p_start,
    const Vector3d& p_goal,
    const RotMatrix3d& R_target,
    const std::vector<ObstacleInfo>& obstacles) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obstacles;
    return planImpl(q_start, q_goal, {q_goal}, 0);
}

PlanResult AapfBiRRTStar::planImpl(
    const JointConfig& q_start,
    const JointConfig& q_goal,
    const std::vector<JointConfig>& requested_goal_candidates,
    unsigned int request_seed,
    const PlanRequestCore* request) {
    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0 ? 7U : request_seed));
    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;
    SearchRuntime runtime(request, t_start);

    if (!collision_) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT* requires a collision checker.";
        return result;
    }

    if (!PathValidator::finite(q_start) || !PathValidator::finite(q_goal) ||
        !limits_.isWithin(q_start, kEpsJointLimitTol) ||
        !limits_.isWithin(q_goal, kEpsJointLimitTol)) {
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message = "AAPF-BiRRT*: non-finite or out-of-limit request state.";
        return result;
    }

    const std::vector<JointConfig> fallback_goal_candidates{q_goal};
    const std::vector<JointConfig>& goal_candidates = requested_goal_candidates.empty()
        ? fallback_goal_candidates : requested_goal_candidates;
    if (goal_candidates.empty()) {
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "AAPF-BiRRT*: requested goal joint target is invalid or in collision.";
        return result;
    }
    for (const auto& candidate : goal_candidates) {
        if (!PathValidator::finite(candidate) || !limits_.isWithin(candidate, kEpsJointLimitTol) ||
            !collision_->isStateValid(candidate)) {
            result.failure_code = PlanningFailureCode::kGoalNotReached;
            result.message = "AAPF-BiRRT*: requested goal joint target is invalid or in collision.";
            return result;
        }
    }

    if (!collision_->isStateValid(q_start)) {
        result.failure_code = PlanningFailureCode::kCollision;
        result.message = "AAPF-BiRRT*: start joint target is in collision.";
        return result;
    }

    if (params_.aapf.enable && !collision_->supportsWorldClearance()) {
        result.failure_code = PlanningFailureCode::kInvalidInput;
        result.message =
            "AAPF-BiRRT*: collision backend does not support world-clearance gradients.";
        return result;
    }
    const double path_validation_distance =
        std::max(kEpsJointNear, params_.validation_distance);
    const double strict_validation_distance = std::min(
        path_validation_distance,
        std::max(kEpsJointNear, params_.aapf.strict_validation_distance));

    const PathValidator validator{*collision_, path_validation_distance, strict_validation_distance};

    std::vector<JointConfig> direct_path;
    for (const auto& candidate : goal_candidates) {
        if (!validator.strict(q_start, candidate)) continue;
        const std::vector<JointConfig> candidate_path{q_start, candidate};
        if (direct_path.empty() || PathValidator::cost(candidate_path) < PathValidator::cost(direct_path)) {
            direct_path = candidate_path;
        }
    }

    const int max_n = params_.max_iterations * 3 + 64;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    for (const auto& gc : goal_candidates) {
        treeB.addNode(gc, -1, 0.0);
    }

    AapfGuidedSampler aapf_sampler(
        params_, limits_, *collision_, validator.basic_distance, rng_);

    std::array<AapfAdaptiveSamplerSelector, 2> selectors{
        AapfAdaptiveSamplerSelector(
            params_.aapf.adaptive_window, params_.aapf.adaptive_exploration,
            params_.aapf.global_min_period),
        AapfAdaptiveSamplerSelector(
            params_.aapf.adaptive_window, params_.aapf.adaptive_exploration,
            params_.aapf.global_min_period),
    };
    std::array<std::array<SamplingStats, kAapfSampleSourceCount>, 2> sampling_stats{};
    for (int tree = 0; tree < 2; ++tree) {
        for (int source = 0; source < kAapfSampleSourceCount; ++source) {
            sampling_stats[tree][source].tree_index = tree;
            sampling_stats[tree][source].source = aapfSampleSourceName(
                static_cast<AapfSampleSource>(source));
        }
    }
    std::vector<JointConfig> best_path = direct_path;
    double best_path_cost = direct_path.empty()
        ? std::numeric_limits<double>::infinity()
        : PathValidator::cost(direct_path);
    int first_goal_sample_attempt = direct_path.empty() ? -1 : 0;
    const int post_solution_budget = std::max(0, params_.post_solution_sample_attempts);
    int iterations_completed = 0;
    std::array<int, 2> last_progress_iter{0, 0};
    if (!direct_path.empty()) {
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = best_path_cost;
    }
    bool grow_a = true;

    const auto considerConnection = [&](int conn_a, int conn_b, int iteration) {
        std::vector<JointConfig> candidate_path;
        if (!buildConnectedPath(treeA, conn_a, treeB, conn_b, validator, &candidate_path)) {
            return ConnectionQuality{};
        }
        const double candidate_cost = PathValidator::cost(candidate_path);
        ConnectionQuality quality;
        quality.valid = true;
        if (candidate_cost + kEpsCostEqual < best_path_cost) {
            best_path = std::move(candidate_path);
            best_path_cost = candidate_cost;
            if (first_goal_sample_attempt < 0) {
                first_goal_sample_attempt = result.sample_attempts;
                result.first_solution_time_s = runtime.elapsedSeconds();
                result.first_solution_path_cost = best_path_cost;
            }
            quality.improved = true;
        }
        return quality;
    };

    const auto saveSamplingStats = [&]() {
        result.sampling_stats.clear();
        std::ostringstream diagnostics;
        diagnostics << "AAPF adaptive sampling:";
        for (const auto& by_source : sampling_stats) {
            for (const auto& stats : by_source) {
                result.sampling_stats.push_back(stats);
                diagnostics << " tree=" << stats.tree_index
                            << " source=" << stats.source
                            << " selected=" << stats.selections
                            << " proposed=" << stats.proposals
                            << " inserted=" << stats.insertions
                            << " progressed=" << stats.progress_events
                            << " connected=" << stats.connections
                            << " improved=" << stats.improvements
                            << " clearance=" << stats.clearance_queries
                            << " utility=" << std::fixed << std::setprecision(3)
                            << stats.utility_sum
                            << " elapsed_ms=" << std::setprecision(1) << stats.elapsed_ms << ';';
            }
        }
        result.diagnostics = diagnostics.str();
    };

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (runtime.shouldStop()) break;
        if (post_solution_budget > 0 && first_goal_sample_attempt >= 0 &&
            result.sample_attempts - first_goal_sample_attempt >= post_solution_budget) {
            result.effort_stats.post_solution_budget_complete = true;
            break;
        }
        runtime.capture(result, !best_path.empty(), best_path_cost, treeA.size() + treeB.size());
        iterations_completed = it;
        const int tree_index = grow_a ? 0 : 1;
        ++result.sample_attempts;
        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;
        const AapfSampleSource requested_source = params_.aapf.enable
            ? selectors[tree_index].choose()
            : AapfSampleSource::kGlobal;
        SamplingStats& source_stats =
            sampling_stats[tree_index][aapfSampleSourceIndex(requested_source)];
        ++source_stats.selections;
        const auto iteration_start = std::chrono::steady_clock::now();
        JointConfig q_target = q_start;
        if (grow_a) {
            double best_target_distance = std::numeric_limits<double>::infinity();
            for (const auto& candidate : goal_candidates) {
                const int nearest = cur.nearest(candidate);
                const double distance = jointDistance(cur.node(nearest).state, candidate);
                if (distance < best_target_distance) {
                    best_target_distance = distance;
                    q_target = candidate;
                }
            }
        }
        AapfGuidedSample step = aapf_sampler.generate(
            cur, opp, q_target, tree_index,
            std::max(0, it - last_progress_iter[tree_index]), requested_source);
        source_stats.clearance_queries += step.clearance_queries;
        if (step.proposal_generated) ++source_stats.proposals;
        bool valid_extension = step.valid_edge || step.basic_edge_prevalidated;
        if (!valid_extension && step.proposal_generated &&
            requested_source == AapfSampleSource::kGlobal) {
            valid_extension = validator.basic(step.q_near, step.q_new);
        }
        if (!valid_extension && step.proposal_generated &&
            requested_source == AapfSampleSource::kGlobal) {
            JointConfig q_shrunk;
            double shrink_dist = 0.0;
            if (shrinkMotionToward(
                    params_, limits_, *collision_, validator.basic_distance,
                    step.q_near, step.q_new, &q_shrunk, &shrink_dist)) {
                step.q_new = q_shrunk;
                valid_extension = true;
            }
        }
        if (!valid_extension) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - iteration_start).count();
            source_stats.elapsed_ms += elapsed_ms;
            selectors[tree_index].observe(requested_source, 0.0, elapsed_ms);
            grow_a = !grow_a;
            continue;
        }

        const bool enable_rewire =
            params_.rewire_every_k > 0 &&
            (it % std::max(1, params_.rewire_every_k) == 0);
        const int new_idx = extendRrtStar(
            cur, step.q_new, step.idx_near, params_, *collision_,
            validator.basic_distance, true, enable_rewire);
        if (new_idx >= 0) {
            ++source_stats.insertions;
            ++result.accepted_samples;
        }

        const int before_idx = opp.nearest(step.q_near);
        const int after_idx = new_idx >= 0 ? opp.nearest(step.q_new) : -1;
        const double distance_before = before_idx < 0 ? 0.0 :
            jointDistance(step.q_near, opp.node(before_idx).state);
        const double distance_after = after_idx < 0 ? distance_before :
            jointDistance(step.q_new, opp.node(after_idx).state);
        const double progress = std::clamp(
            (distance_before - distance_after) / std::max(kEpsJointNear, params_.max_step),
            0.0, 1.0);
        ConnectionQuality quality;
        if (new_idx >= 0) {
            ConnectionResult connection = tryConnect(
                params_, limits_, *collision_, validator.basic_distance,
                step.q_new, opp);
            if (connection.connected) {
                bool inserted = false;
                const int cur_conn = appendConnectionBridge(
                    cur, new_idx, connection, validator, params_, &inserted);
                if (cur_conn >= 0) {
                    quality = considerConnection(
                        grow_a ? cur_conn : connection.idx_other,
                        grow_a ? connection.idx_other : cur_conn, it);
                }
            } else if (connection.advanced) {
                const int bridge_idx = appendConnectionBridge(
                    cur, new_idx, connection, validator, params_);
                if (bridge_idx >= 0) {
                    ConnectionResult retry = tryConnect(
                        params_, limits_, *collision_, validator.basic_distance,
                        cur.node(bridge_idx).state, opp);
                    if (retry.connected) {
                        const int cur_conn = appendConnectionBridge(
                            cur, bridge_idx, retry, validator, params_);
                        if (cur_conn >= 0) {
                            quality = considerConnection(
                                grow_a ? cur_conn : retry.idx_other,
                                grow_a ? retry.idx_other : cur_conn, it);
                        }
                    }
                }
            }
        }
        if (quality.valid) ++source_stats.connections;
        if (quality.improved) ++source_stats.improvements;
        if (progress > kEpsCostEqual) ++source_stats.progress_events;
        if (progress > kEpsCostEqual || quality.valid || quality.improved) {
            last_progress_iter[tree_index] = it;
        }
        const double utility = std::clamp(
            (new_idx >= 0 ? 0.25 : 0.0) + 0.50 * progress +
                (quality.valid ? 0.25 : 0.0),
            0.0, 1.0);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - iteration_start).count();
        source_stats.utility_sum += utility;
        source_stats.elapsed_ms += elapsed_ms;
        selectors[tree_index].observe(requested_source, utility, elapsed_ms);
        grow_a = !grow_a;
    }

    if (best_path.empty()) {
        result.success = false;
        result.failure_code = runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached;
        result.iterations = iterations_completed;
        result.planning_time = runtime.elapsedSeconds();

        result.message = runtime.deadlineReached() ? "deadline_no_solution" : "AAPF-BiRRT* failed to connect trees within iteration guard.";
        runtime.markStopReason(result, iterations_completed >= params_.max_iterations);
        saveSamplingStats();
        return result;
    }

    if (!validator.finalize(&best_path)) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kCollision;
        result.iterations = iterations_completed;
        result.planning_time = runtime.elapsedSeconds();
        result.message = "AAPF-BiRRT*: final independent path validation failed.";
        saveSamplingStats();
        return result;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.path = std::move(best_path);
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_path_cost;
    if (first_goal_sample_attempt >= 0) {
        result.effort_stats.post_solution_sample_attempts = std::max(
            0, result.sample_attempts - first_goal_sample_attempt);
    }
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = iterations_completed;
    runtime.capture(result, true, result.path_cost, result.num_nodes);
    runtime.markStopReason(result, iterations_completed >= params_.max_iterations);
    saveSamplingStats();
    return result;
}

}  // namespace fairino_planning
