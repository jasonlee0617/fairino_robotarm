// src/algorithms/bidirectional/bi_rrt_star.cpp
// ponytail: minimal BiRRT* baseline without IK-guided sampling.
#include "myrobot_planning_core/algorithms/bi_rrt_star.h"
#include "../common/bidirectional_utils.hpp"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fairino_planning {
namespace {

constexpr double kEpsCostEqual = 1e-12;
constexpr double kEpsPathDedup = 1e-10;
constexpr double kEpsJointLimitTol = 1e-4;

JointConfig jointDelta(const JointConfig& from, const JointConfig& to) { return to - from; }
double jointDistance(const JointConfig& a, const JointConfig& b) { return jointDelta(a, b).norm(); }
double jointDistanceSq(const JointConfig& a, const JointConfig& b) { return jointDelta(a, b).squaredNorm(); }

JointConfig sampleUniform(const JointLimits& limits, std::mt19937& rng) {
    return limits.sampleUniform(rng);
}

}  // namespace

std::vector<ObstacleInfo> BiRRTStar::normalizeObstacles(
    const Vector3d& obs_origin,
    const Vector3d& obs_size,
    const std::vector<ObstacleInfo>& obstacles) {
    return bidirectional::normalizeObstacles(obs_origin, obs_size, obstacles);
}

BiRRTStar::BiRRTStar() : rng_(7) {}

PlanResult BiRRTStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    const auto obstacles = normalizeObstacles(request.obs_origin, request.obs_size, request.obstacles);
    return planImpl(request.q_start,
        request.goal_candidates.empty() ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates,
        obstacles, request.random_seed, &request);
}

PlanResult BiRRTStar::plan(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const Vector3d& obs_origin, const Vector3d& obs_size) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    return planImpl(q_start, {q_goal}, normalizeObstacles(obs_origin, obs_size, {}), 0);
}

PlanResult BiRRTStar::planMultiObs(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const std::vector<ObstacleInfo>& obstacles) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    return planImpl(q_start, {q_goal}, obstacles, 0);
}

double BiRRTStar::computeRewireRadius(int n) const {
    return bidirectional::rewireRadius(params_, n);
}

BiRRTStar::ConnResult BiRRTStar::tryConnect(
    const JointConfig& q_new, RRTTree& other_tree) const {
    ConnResult res;
    int idx_other = other_tree.nearest(q_new);
    res.idx_other = idx_other;
    JointConfig q_near = other_tree.node(idx_other).state;
    double d = jointDistance(q_new, q_near);

    if (d < params_.max_step * params_.direct_connect_step_factor) {
        if (collision_->isMotionValid(q_new, q_near, params_.validation_distance)) {
            res.connected = true;
        }
    } else if (d < params_.max_step * params_.connect_max_steps) {
        JointConfig q_curr = q_new;
        JointConfig q_prev = q_new;
        for (int cs = 0; cs < params_.connect_max_steps; ++cs) {
            JointConfig q_step = limits_.clamp(steer(q_curr, q_near, params_.max_step));
            if (!collision_->isStateValid(q_step)) break;
            if (!collision_->isMotionValid(q_prev, q_step, params_.validation_distance)) break;
            res.bridge.push_back(q_step);
            if (jointDistance(q_step, q_near) < params_.connect_target_tolerance) {
                if (collision_->isMotionValid(q_step, q_near, params_.validation_distance)) {
                    res.connected = true;
                }
                break;
            }
            q_prev = q_step;
            q_curr = q_step;
        }
        if (!res.connected && !res.bridge.empty()) {
            const JointConfig& q_last = res.bridge.back();
            if (jointDistance(q_last, q_near) < params_.max_step * params_.direct_connect_step_factor) {
                if (collision_->isMotionValid(q_last, q_near, params_.validation_distance)) {
                    res.connected = true;
                }
            }
        }
    }
    return res;
}

PlanResult BiRRTStar::planImpl(
    const JointConfig& q_start,
    const std::vector<JointConfig>& requested_goal_candidates,
    const std::vector<ObstacleInfo>& obstacles,
    unsigned int request_seed,
    const PlanRequestCore* request) {
    (void)obstacles;

    PlanResult fast_fail;
    if (!collision_) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "BiRRT*: null collision checker.";
        return fast_fail;
    }
    if (requested_goal_candidates.empty()) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "BiRRT*: no goal candidates.";
        return fast_fail;
    }
    const std::vector<JointConfig>& goal_candidates = requested_goal_candidates;
    if (!limits_.isWithin(q_start, kEpsJointLimitTol)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "BiRRT*: start or goal out of joint limits.";
        return fast_fail;
    }
    if (!bidirectional::isFinite(q_start)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
        fast_fail.message = "BiRRT*: start or goal contains NaN.";
        return fast_fail;
    }
    if (!collision_->isStateValid(q_start)) {
        fast_fail.success = false;
        fast_fail.failure_code = PlanningFailureCode::kGoalNotReached;
        fast_fail.message = "BiRRT*: start configuration in collision.";
        return fast_fail;
    }
    for (const auto& q_goal : goal_candidates) {
        if (!limits_.isWithin(q_goal, kEpsJointLimitTol) || !bidirectional::isFinite(q_goal)) {
            fast_fail.success = false;
            fast_fail.failure_code = PlanningFailureCode::kInvalidInput;
            fast_fail.message = "BiRRT*: goal configuration is non-finite or out of joint limits.";
            return fast_fail;
        }
        if (!collision_->isStateValid(q_goal)) {
            fast_fail.success = false;
            fast_fail.failure_code = PlanningFailureCode::kGoalNotReached;
            fast_fail.message = "BiRRT*: goal configuration in collision.";
            return fast_fail;
        }
    }

    rng_.seed(static_cast<std::mt19937::result_type>(request_seed == 0 ? 7U : request_seed));

    auto t_start = std::chrono::steady_clock::now();
    PlanResult result;
    SearchRuntime runtime(request, t_start);

    const int max_n = params_.max_iterations / 2 + 10;
    RRTTree treeA(max_n), treeB(max_n);
    treeA.addNode(q_start, -1, 0.0);
    for (const auto& q_goal : goal_candidates) treeB.addNode(q_goal, -1, 0.0);

    double best_cost = std::numeric_limits<double>::infinity();
    int best_conn_a = -1, best_conn_b = -1;
    int first_goal_sample_attempt = -1;
    const int post_solution_budget = std::max(0, params_.post_solution_sample_attempts);
    int connect_every_k = 1;
    bool grow_a = true;
    int iterations_completed = 0;

    for (int goal_index = 0; goal_index < static_cast<int>(goal_candidates.size()); ++goal_index) {
        const double direct_cost = jointDistance(q_start, goal_candidates[goal_index]);
        if (direct_cost + kEpsCostEqual >= best_cost ||
            !collision_->isMotionValid(q_start, goal_candidates[goal_index], params_.validation_distance)) {
            continue;
        }
        best_cost = direct_cost;
        best_conn_a = 0;
        best_conn_b = goal_index;
    }
    if (best_conn_a >= 0) {
        first_goal_sample_attempt = 0;
        result.first_solution_time_s = runtime.elapsedSeconds();
        result.first_solution_path_cost = best_cost;
    }

    for (int it = 1; it <= params_.max_iterations; ++it) {
        if (runtime.shouldStop()) break;
        if (post_solution_budget > 0 && first_goal_sample_attempt >= 0 &&
            result.sample_attempts - first_goal_sample_attempt >= post_solution_budget) {
            result.effort_stats.post_solution_budget_complete = true;
            break;
        }
        runtime.capture(result, std::isfinite(best_cost), best_cost, treeA.size() + treeB.size());
        iterations_completed = it;
        ++result.sample_attempts;

        RRTTree& cur = grow_a ? treeA : treeB;
        RRTTree& opp = grow_a ? treeB : treeA;
        JointConfig q_rand = sampleUniform(limits_, rng_);
        if (params_.goal_bias > 0.0) {
            std::uniform_real_distribution<double> goal_coin(0.0, 1.0);
            if (goal_coin(rng_) < params_.goal_bias) {
                std::uniform_int_distribution<int> goal_idx(0, opp.size() - 1);
                q_rand = opp.node(goal_idx(rng_)).state;
            }
        }
        int idx_near = cur.nearest(q_rand);
        JointConfig q_near = cur.node(idx_near).state;
        JointConfig q_new = limits_.clamp(steer(q_near, q_rand, params_.max_step));

        if (!collision_->isStateValid(q_new)) { grow_a = !grow_a; continue; }

        double rr = computeRewireRadius(cur.size());
        auto near_set = cur.nearRadius(q_new, rr);
        if (near_set.empty()) near_set.push_back(idx_near);
        if (static_cast<int>(near_set.size()) > params_.max_near) {
            std::partial_sort(near_set.begin(),
                near_set.begin() + params_.max_near, near_set.end(),
                [&](int a, int b) {
                    return jointDistanceSq(cur.node(a).state, q_new) <
                           jointDistanceSq(cur.node(b).state, q_new);
                });
            near_set.resize(params_.max_near);
        }

        struct Cand { int idx; double cost; };
        std::vector<Cand> cands;
        for (int ic : near_set) {
            cands.push_back({ic, cur.node(ic).cost + jointDistance(cur.node(ic).state, q_new)});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

        int best_par = -1;
        double best_c2n = std::numeric_limits<double>::infinity();
        for (auto& c : cands) {
            if (collision_->isMotionValid(cur.node(c.idx).state, q_new, params_.validation_distance)) {
                best_par = c.idx; best_c2n = c.cost; break;
            }
        }
        if (best_par < 0) {
            if (!collision_->isMotionValid(q_near, q_new, params_.validation_distance)) {
                grow_a = !grow_a; continue;
            }
            best_par = idx_near;
            best_c2n = cur.node(idx_near).cost + jointDistance(q_near, q_new);
        }

        int new_idx = cur.addNode(q_new, best_par, best_c2n);
        ++result.accepted_samples;

        if (it % params_.rewire_every_k == 0) {
            int rw_n = std::min(params_.rewire_max_neighbors, static_cast<int>(near_set.size()));
            for (int kk = 0; kk < rw_n; ++kk) {
                int j = near_set[kk];
                if (j == best_par || j == new_idx) continue;
                double ej = jointDistance(q_new, cur.node(j).state);
                double cvn = cur.node(new_idx).cost + ej;
                if (cvn + kEpsCostEqual >= cur.node(j).cost) continue;
                if (!collision_->isMotionValid(q_new, cur.node(j).state, params_.validation_distance)) continue;
                cur.reparent(j, new_idx, cvn);
            }
        }

        // Two-phase bridge: validate the full new_idx→bridge→other chain in local
        // variables first, then insert nodes only when total < best_cost.
        if (it % connect_every_k == 0) {
            auto conn = tryConnect(q_new, opp);
            if (conn.connected) {
                // Phase 1: validate every bridge segment and compute final cost
                // without mutating the tree.
                double chain_cost = cur.node(new_idx).cost;
                JointConfig q_bridge_prev = q_new;
                bool bridge_valid = true;
                for (const auto& q_bridge : conn.bridge) {
                    if (!collision_->isMotionValid(q_bridge_prev, q_bridge, params_.validation_distance)) {
                        bridge_valid = false; break;
                    }
                    chain_cost += jointDistance(q_bridge_prev, q_bridge);
                    q_bridge_prev = q_bridge;
                }
                if (!bridge_valid) { grow_a = !grow_a; continue; }

                const JointConfig& q_bridge_last = conn.bridge.empty() ? q_new : conn.bridge.back();
                const JointConfig& q_other = opp.node(conn.idx_other).state;
                if (!collision_->isMotionValid(q_bridge_last, q_other, params_.validation_distance)) {
                    grow_a = !grow_a; continue;
                }
                double total = chain_cost +
                    jointDistance(q_bridge_last, q_other) +
                    opp.node(conn.idx_other).cost;

                if (total < best_cost) {
                    // Phase 2: insert bridge nodes now that we know it's a
                    // strict improvement.
                    int bridge_parent = new_idx;
                    for (const auto& q_bridge : conn.bridge) {
                        double bridge_cost = cur.node(bridge_parent).cost +
                            jointDistance(cur.node(bridge_parent).state, q_bridge);
                        bridge_parent = cur.addNode(q_bridge, bridge_parent, bridge_cost);
                    }
                    best_cost = total;
                    best_conn_a = grow_a ? bridge_parent : conn.idx_other;
                    best_conn_b = grow_a ? conn.idx_other : bridge_parent;
                    if (first_goal_sample_attempt < 0) {
                        first_goal_sample_attempt = result.sample_attempts;
                        result.first_solution_time_s = runtime.elapsedSeconds();
                        result.first_solution_path_cost = best_cost;
                    }
                    connect_every_k = params_.connect_success_every_k;
                }
            }
        }

        grow_a = !grow_a;
    }

    if (first_goal_sample_attempt >= 0) {
        result.effort_stats.post_solution_sample_attempts = std::max(
            0, result.sample_attempts - first_goal_sample_attempt);
    }

    if (best_conn_a < 0) {
        result.success = false;
        result.failure_code = runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached;
        result.message = runtime.deadlineReached() ? "deadline_no_solution" : "BiRRT* failed: no connection found.";
        result.iterations = iterations_completed;
        result.planning_time = runtime.elapsedSeconds();
        runtime.markStopReason(result, iterations_completed >= params_.max_iterations);
        return result;
    }

    auto pathA = treeA.backtrack(best_conn_a);
    auto pathB = treeB.backtrack(best_conn_b);
    std::reverse(pathB.begin(), pathB.end());
    result.path.clear();
    result.path.insert(result.path.end(), pathA.begin(), pathA.end());
    result.path.insert(result.path.end(), pathB.begin(), pathB.end());
    auto it_dup = std::unique(result.path.begin(), result.path.end(),
        [](const JointConfig& a, const JointConfig& b) {
            return (a - b).norm() < kEpsPathDedup;
        });
    result.path.erase(it_dup, result.path.end());

    int bad_seg = -1;
    if (!bidirectional::strictlyValid(*collision_, result.path, params_.validation_distance, &bad_seg)) {
        result.success = false;
        result.failure_code = PlanningFailureCode::kGoalNotReached;
        result.message = "BiRRT* final path invalid at segment " + std::to_string(bad_seg);
        return result;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.success = true;
    result.failure_code = PlanningFailureCode::kNone;
    result.planning_time = std::chrono::duration<double>(t_end - t_start).count();
    result.path_cost = best_cost;
    result.num_nodes = treeA.size() + treeB.size();
    result.iterations = iterations_completed;
    runtime.capture(result, true, result.path_cost, result.num_nodes);
    runtime.markStopReason(result, iterations_completed >= params_.max_iterations);
    return result;
}

}  // namespace fairino_planning
