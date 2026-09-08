#include "myrobot_planning_core/algorithms/mire_bi_ait_star.h"
#include "myrobot_planning_core/algorithms/search_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>

#include "../common/goal_set_utils.hpp"
#include "../common/informed_goal_sampler.hpp"

namespace fairino_planning {
namespace {

constexpr double kEps = 1e-10;
constexpr int kInfiniteEffort = std::numeric_limits<int>::max() / 4;

bool isFinite(const JointConfig& q) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (!std::isfinite(q[i])) return false;
    }
    return true;
}

double distance(const JointConfig& a, const JointConfig& b) {
    return (a - b).norm();
}

std::uint64_t edgeKey(int a, int b) {
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U) | hi;
}

enum class EdgeState : unsigned char { kUnknown, kFree, kBlocked };

struct EdgeInfo {
    EdgeState state = EdgeState::kUnknown;
    bool midpoint_checked = false;
    bool quarters_checked = false;
};

struct CostEffort {
    double cost = std::numeric_limits<double>::infinity();
    int effort = kInfiniteEffort;
    int root = -1;
};

bool better(const CostEffort& lhs, const CostEffort& rhs) {
    if (lhs.cost + kEps < rhs.cost) return true;
    if (rhs.cost + kEps < lhs.cost) return false;
    if (lhs.effort != rhs.effort) return lhs.effort < rhs.effort;
    return lhs.root >= 0 && (rhs.root < 0 || lhs.root < rhs.root);
}

bool same(const CostEffort& lhs, const CostEffort& rhs) {
    const bool same_cost = (std::isinf(lhs.cost) || std::isinf(rhs.cost))
        ? lhs.cost == rhs.cost
        : std::abs(lhs.cost - rhs.cost) <= kEps;
    return same_cost && lhs.effort == rhs.effort &&
           lhs.root == rhs.root;
}

CostEffort minimum(const CostEffort& lhs, const CostEffort& rhs) {
    return better(lhs, rhs) ? lhs : rhs;
}

struct ReverseState {
    CostEffort g;
    CostEffort rhs;
    std::uint64_t revision = 0;
    bool queued = false;
    bool processing = false;
};

struct ReverseEntry {
    int vertex = -1;
    CostEffort key;
    std::uint64_t revision = 0;
};

enum class SampleOrigin : unsigned char {
    kTerminal,
    kUniform,
    kInformed,
};

struct ReverseCompare {
    bool operator()(const ReverseEntry& lhs, const ReverseEntry& rhs) const {
        return better(rhs.key, lhs.key);
    }
};

struct ForwardNode {
    double cost = std::numeric_limits<double>::infinity();
    int parent = -1;
    int generation = 0;
    int root = -1;
};

struct ForwardEntry {
    int from = -1;
    int to = -1;
    int generation = 0;
    double lower_bound = std::numeric_limits<double>::infinity();
    std::uint64_t serial = 0;
};

struct ForwardCompare {
    bool operator()(const ForwardEntry& lhs, const ForwardEntry& rhs) const {
        if (std::abs(lhs.lower_bound - rhs.lower_bound) > kEps) {
            return lhs.lower_bound > rhs.lower_bound;
        }
        return lhs.serial > rhs.serial;
    }
};

struct PostSearchEntry {
    int vertex = -1;
    double joint_lower_bound = std::numeric_limits<double>::infinity();
    double tcp_lower_bound = std::numeric_limits<double>::infinity();
    std::uint64_t serial = 0;
};

struct PostSearchCompare {
    bool operator()(const PostSearchEntry& lhs, const PostSearchEntry& rhs) const {
        if (std::abs(lhs.joint_lower_bound - rhs.joint_lower_bound) > kEps) {
            return lhs.joint_lower_bound > rhs.joint_lower_bound;
        }
        if (std::abs(lhs.tcp_lower_bound - rhs.tcp_lower_bound) > kEps) {
            return lhs.tcp_lower_bound > rhs.tcp_lower_bound;
        }
        return lhs.serial > rhs.serial;
    }
};

std::vector<int> rootSampleSchedule(
    const JointConfig& start, const std::vector<JointConfig>& goals,
    double best_cost, int budget) {
    struct Allocation {
        int root = -1;
        int quota = 0;
        double gap = 0.0;
        double remainder = 0.0;
    };
    std::vector<Allocation> allocations;
    for (size_t root = 0; root < goals.size(); ++root) {
        const double gap = best_cost - distance(start, goals[root]);
        if (gap > kEps) allocations.push_back({static_cast<int>(root), 0, gap, 0.0});
    }
    if (budget <= 0 || allocations.empty()) return {};

    int assigned = 0;
    for (auto& allocation : allocations) {
        if (assigned == budget) break;
        allocation.quota = 1;
        ++assigned;
    }
    const int remaining = budget - assigned;
    if (remaining > 0) {
        double total_gap = 0.0;
        for (const auto& allocation : allocations) total_gap += allocation.gap;
        int distributed = 0;
        for (auto& allocation : allocations) {
            const double exact = remaining * allocation.gap / total_gap;
            const int whole = static_cast<int>(std::floor(exact));
            allocation.quota += whole;
            allocation.remainder = exact - whole;
            distributed += whole;
        }
        std::vector<size_t> order(allocations.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            const double delta = allocations[lhs].remainder - allocations[rhs].remainder;
            if (std::abs(delta) > kEps) return delta > 0.0;
            return allocations[lhs].root < allocations[rhs].root;
        });
        for (int i = distributed; i < remaining; ++i) {
            ++allocations[order[static_cast<size_t>(i - distributed)]].quota;
        }
    }

    std::vector<int> schedule;
    schedule.reserve(static_cast<size_t>(budget));
    while (static_cast<int>(schedule.size()) < budget) {
        for (auto& allocation : allocations) {
            if (allocation.quota <= 0) continue;
            schedule.push_back(allocation.root);
            --allocation.quota;
        }
    }
    return schedule;
}

}  // namespace

PlanResult MireBiAitStar::plan(const PlanRequestCore& request) {
    setToolModel(request.tool_model);
    PlanResult result;
    const auto started = std::chrono::steady_clock::now();
    SearchRuntime runtime(&request, started);
    const auto finish = [&](bool success, PlanningFailureCode code, const std::string& message) {
        result.success = success;
        result.failure_code = code;
        result.message = message;
        result.sample_attempts = result.effort_stats.sample_attempts;
        result.accepted_samples = result.effort_stats.sampled_states;
        result.lower_bound_certified = result.effort_stats.lower_bound_certified;
        result.first_solution_time_s = result.effort_stats.first_solution_time_s;
        result.first_solution_path_cost = result.effort_stats.first_solution_path_cost;
        result.planning_time = runtime.elapsedSeconds();
        runtime.capture(result, success, result.path_cost, result.num_nodes);
        runtime.markStopReason(result, result.effort_stats.work_units >= std::max(1, params_.max_iterations));
        return result;
    };

    if (!collision_) {
        return finish(false, PlanningFailureCode::kInvalidInput,
            "MIRE-BiAIT*: null collision checker.");
    }
    if (!isFinite(request.q_start) || !limits_.isWithin(request.q_start)) {
        return finish(false, PlanningFailureCode::kInvalidInput,
            "MIRE-BiAIT*: start configuration is non-finite or out of joint limits.");
    }

    const std::vector<JointConfig> requested_goals = request.goal_candidates.empty()
        ? std::vector<JointConfig>{request.q_goal} : request.goal_candidates;
    std::vector<JointConfig> goals;
    for (const auto& goal : requested_goals) {
        if (!isFinite(goal) || !limits_.isWithin(goal)) {
            return finish(false, PlanningFailureCode::kInvalidInput,
                "MIRE-BiAIT*: goal configuration is non-finite or out of joint limits.");
        }
        if (std::none_of(goals.begin(), goals.end(), [&](const JointConfig& existing) {
                return distance(existing, goal) < kEps;
            })) {
            goals.push_back(goal);
        }
    }
    if (goals.empty()) {
        return finish(false, PlanningFailureCode::kInvalidInput,
            "MIRE-BiAIT*: no goal candidates.");
    }
    const double goal_lower_bound = goal_set::lowerBound(request.q_start, goals);

    const auto validState = [&](const JointConfig& q) {
        ++result.effort_stats.state_validation_calls;
        return collision_->isStateValid(q);
    };
    if (!validState(request.q_start)) {
        return finish(false, PlanningFailureCode::kGoalNotReached,
            "MIRE-BiAIT*: start configuration in collision.");
    }
    for (const auto& goal : goals) {
        if (!validState(goal)) {
            return finish(false, PlanningFailureCode::kGoalNotReached,
                "MIRE-BiAIT*: goal configuration in collision.");
        }
    }

    const int max_work = std::max(1, params_.max_iterations);
    const int batch_size = std::max(1, params_.mire_biait.batch_size);
    const int post_batch_size = std::max(1, params_.mire_biait.post_solution_batch_size);
    const int post_solution_budget = std::max(0, params_.post_solution_sample_attempts);
    const double validation_distance = std::max(1e-4, params_.validation_distance);
    const double focal_factor = params_.mire_biait.enable_effort_focal_queue
        ? std::max(1.0, params_.mire_biait.effort_focal_factor) : 1.0;
    const unsigned int effective_seed = request.random_seed == 0U ? 7U : request.random_seed;
    std::mt19937 rng(static_cast<std::mt19937::result_type>(effective_seed));
    const auto consumeWork = [&]() {
        if (runtime.shouldStop()) return false;
        if (result.effort_stats.work_units >= max_work) {
            result.effort_stats.budget_exhausted = true;
            return false;
        }
        result.iterations = ++result.effort_stats.work_units;
        if (result.effort_stats.work_units == max_work) {
            result.effort_stats.budget_exhausted = true;
        }
        return true;
    };

    std::vector<JointConfig> vertices;
    std::vector<Vector3d> tcp_positions;
    std::vector<std::vector<int>> adjacency;
    std::vector<SampleOrigin> vertex_origins;
    vertices.reserve(static_cast<size_t>(max_work + goals.size() + 1));
    tcp_positions.reserve(vertices.capacity());
    adjacency.reserve(vertices.capacity());
    vertex_origins.reserve(vertices.capacity());
    vertices.push_back(request.q_start);
    tcp_positions.push_back(fk_.fkine(request.q_start, request.tool_model).block<3, 1>(0, 3));
    adjacency.emplace_back();
    vertex_origins.push_back(SampleOrigin::kTerminal);
    for (const auto& goal : goals) {
        vertices.push_back(goal);
        tcp_positions.push_back(fk_.fkine(goal, request.tool_model).block<3, 1>(0, 3));
        adjacency.emplace_back();
        vertex_origins.push_back(SampleOrigin::kTerminal);
    }
    const int start_index = 0;
    std::vector<int> roots;
    roots.reserve(goals.size());
    for (size_t i = 0; i < goals.size(); ++i) roots.push_back(static_cast<int>(i + 1U));
    result.effort_stats.goal_root_count = static_cast<int>(roots.size());
    result.effort_stats.root_lower_bounds.reserve(goals.size());
    for (const auto& goal : goals) {
        result.effort_stats.root_lower_bounds.push_back(distance(request.q_start, goal));
    }
    result.effort_stats.root_sample_attempts.assign(goals.size(), 0);
    result.effort_stats.root_accepted_samples.assign(goals.size(), 0);
    result.effort_stats.root_path_improvements.assign(goals.size(), 0);
    const auto isRoot = [&](int index) {
        return index > 0 && index <= static_cast<int>(goals.size());
    };

    std::unordered_map<std::uint64_t, EdgeInfo> edges;
    std::vector<ReverseState> reverse(vertices.size());
    std::priority_queue<ReverseEntry, std::vector<ReverseEntry>, ReverseCompare> reverse_queue;
    std::vector<ForwardNode> forward(vertices.size());
    forward[static_cast<size_t>(start_index)].cost = 0.0;
    forward[static_cast<size_t>(start_index)].generation = 1;
    std::vector<ForwardNode> goal_tree(vertices.size());
    for (const int root : roots) {
        ForwardNode& node = goal_tree[static_cast<size_t>(root)];
        node.cost = 0.0;
        node.generation = 1;
        node.root = root;
    }
    std::priority_queue<ForwardEntry, std::vector<ForwardEntry>, ForwardCompare> forward_queue;
    std::priority_queue<ForwardEntry, std::vector<ForwardEntry>, ForwardCompare> goal_queue;
    std::uint64_t next_forward_serial = 0;
    std::vector<JointConfig> best_path;
    double best_cost = std::numeric_limits<double>::infinity();
    double best_tcp_cost = std::numeric_limits<double>::infinity();
    bool has_solution = false;
    bool post_phase_started = false;

    const auto edgeState = [&](int a, int b) {
        const auto found = edges.find(edgeKey(a, b));
        return found == edges.end() ? EdgeState::kUnknown : found->second.state;
    };
    const auto edgeEffort = [&](int a, int b) {
        const auto found = edges.find(edgeKey(a, b));
        if (found != edges.end() && found->second.state == EdgeState::kBlocked) {
            return kInfiniteEffort;
        }
        if (found != edges.end() && found->second.state == EdgeState::kFree) return 0;
        const int segments = std::max(1, static_cast<int>(std::ceil(
            distance(vertices[static_cast<size_t>(a)], vertices[static_cast<size_t>(b)]) /
            validation_distance)));
        const bool sparse_checked = found != edges.end() && found->second.midpoint_checked;
        return std::max(1, segments - (sparse_checked ? 1 : 0));
    };
    const auto minGoalDistance = [&](int index) {
        double minimum_distance = std::numeric_limits<double>::infinity();
        for (const int root : roots) {
            minimum_distance = std::min(minimum_distance,
                distance(vertices[static_cast<size_t>(index)], vertices[static_cast<size_t>(root)]));
        }
        return minimum_distance;
    };
    std::function<void(int)> updateReverse;
    const auto queueReverse = [&](int vertex) {
        ReverseState& state = reverse[static_cast<size_t>(vertex)];
        if (state.processing || state.queued || same(state.g, state.rhs)) return;
        state.queued = true;
        reverse_queue.push({vertex, minimum(state.g, state.rhs), state.revision});
    };
    updateReverse = [&](int vertex) {
        ReverseState& state = reverse[static_cast<size_t>(vertex)];
        CostEffort rhs;
        if (isRoot(vertex)) {
            rhs = {0.0, 0, vertex};
        } else {
            for (const int next : adjacency[static_cast<size_t>(vertex)]) {
                const EdgeState state_of_edge = edgeState(vertex, next);
                if (state_of_edge == EdgeState::kBlocked) continue;
                const CostEffort& next_label = reverse[static_cast<size_t>(next)].g;
                if (!std::isfinite(next_label.cost) || next_label.effort >= kInfiniteEffort) continue;
                const int effort = edgeEffort(vertex, next);
                if (effort >= kInfiniteEffort) continue;
                rhs = minimum(rhs, {distance(vertices[static_cast<size_t>(vertex)],
                                               vertices[static_cast<size_t>(next)]) + next_label.cost,
                                    std::min(kInfiniteEffort, effort + next_label.effort),
                                    next_label.root});
            }
        }
        if (!same(state.rhs, rhs)) {
            state.rhs = rhs;
            ++state.revision;
        }
        if (!same(state.g, state.rhs)) queueReverse(vertex);
    };

    const auto blockEdge = [&](int a, int b) {
        EdgeInfo& info = edges[edgeKey(a, b)];
        if (info.state == EdgeState::kBlocked) return;
        info.state = EdgeState::kBlocked;
        ++result.effort_stats.blocked_edges;
        if (!post_phase_started) {
            updateReverse(a);
            updateReverse(b);
        }
    };
    const auto markFree = [&](int a, int b) {
        EdgeInfo& info = edges[edgeKey(a, b)];
        if (info.state == EdgeState::kFree) return;
        info.state = EdgeState::kFree;
        if (!post_phase_started) {
            updateReverse(a);
            updateReverse(b);
        }
    };
    const auto sparseCheck = [&](int a, int b) {
        EdgeInfo& info = edges[edgeKey(a, b)];
        if (!params_.mire_biait.enable_lazy_edge_validation) {
            if (info.state == EdgeState::kBlocked) return false;
            if (info.state == EdgeState::kFree) return true;
            ++result.effort_stats.edge_validation_attempts;
            if (collision_->isMotionValid(vertices[static_cast<size_t>(a)],
                                          vertices[static_cast<size_t>(b)],
                                          validation_distance)) {
                markFree(a, b);
                return true;
            }
            blockEdge(a, b);
            return false;
        }
        if (info.state == EdgeState::kBlocked) return false;
        if (info.midpoint_checked) return true;
        info.midpoint_checked = true;
        ++result.effort_stats.sparse_state_checks;
        const JointConfig midpoint = 0.5 * (
            vertices[static_cast<size_t>(a)] + vertices[static_cast<size_t>(b)]);
        if (!validState(midpoint)) {
            blockEdge(a, b);
            return false;
        }
        if (!post_phase_started) {
            updateReverse(a);
            updateReverse(b);
        }
        return true;
    };
    const auto validateFullEdge = [&](int a, int b) {
        const EdgeState state = edgeState(a, b);
        if (state == EdgeState::kFree) return true;
        if (state == EdgeState::kBlocked) return false;
        ++result.effort_stats.edge_validation_attempts;
        if (collision_->isMotionValid(vertices[static_cast<size_t>(a)],
                                      vertices[static_cast<size_t>(b)],
                                      validation_distance)) {
            markFree(a, b);
            return true;
        }
        blockEdge(a, b);
        return false;
    };
    const auto processReverse = [&]() {
        while (!reverse_queue.empty()) {
            const ReverseEntry entry = reverse_queue.top();
            reverse_queue.pop();
            ReverseState& state = reverse[static_cast<size_t>(entry.vertex)];
            state.queued = false;
            if (entry.revision != state.revision ||
                !same(entry.key, minimum(state.g, state.rhs))) {
                ++result.effort_stats.reverse_queue_stale_discards;
                queueReverse(entry.vertex);
                continue;
            }
            if (same(state.g, state.rhs)) {
                ++result.effort_stats.reverse_queue_consistent_discards;
                continue;
            }
            if (!consumeWork()) return;
            state.processing = true;
            ++result.effort_stats.reverse_queue_pops;
            if (better(state.rhs, state.g) && !isRoot(entry.vertex)) {
                int rhs_next = -1;
                CostEffort rhs_candidate;
                for (const int next : adjacency[static_cast<size_t>(entry.vertex)]) {
                    if (edgeState(entry.vertex, next) == EdgeState::kBlocked) continue;
                    const CostEffort& next_label = reverse[static_cast<size_t>(next)].g;
                    if (!std::isfinite(next_label.cost) || next_label.effort >= kInfiniteEffort) continue;
                    const CostEffort candidate{
                        distance(vertices[static_cast<size_t>(entry.vertex)],
                                 vertices[static_cast<size_t>(next)]) + next_label.cost,
                        std::min(kInfiniteEffort, edgeEffort(entry.vertex, next) + next_label.effort),
                        next_label.root};
                    if (rhs_next < 0 || better(candidate, rhs_candidate)) {
                        rhs_candidate = candidate;
                        rhs_next = next;
                    }
                }
                if (rhs_next >= 0 && !sparseCheck(entry.vertex, rhs_next)) {
                    state.processing = false;
                    updateReverse(entry.vertex);
                    return;
                }
            }
            if (better(state.rhs, state.g)) {
                state.g = state.rhs;
            } else {
                state.g = CostEffort{};
            }
            ++state.revision;
            state.processing = false;
            updateReverse(entry.vertex);
            for (const int neighbor : adjacency[static_cast<size_t>(entry.vertex)]) {
                updateReverse(neighbor);
            }
            return;
        }
    };

    const auto enqueueEdge = [&](const std::vector<ForwardNode>& tree, auto& queue,
                                 bool from_start, int from, int to) {
        if (!std::isfinite(tree[static_cast<size_t>(from)].cost)) return;
        const double candidate = tree[static_cast<size_t>(from)].cost +
            distance(vertices[static_cast<size_t>(from)], vertices[static_cast<size_t>(to)]);
        const double lower_bound = from_start
            ? candidate + minGoalDistance(to)
            : distance(vertices[static_cast<size_t>(start_index)],
                       vertices[static_cast<size_t>(to)]) + candidate;
        queue.push({from, to, tree[static_cast<size_t>(from)].generation,
                    lower_bound, next_forward_serial++});
    };
    const auto enqueueFrom = [&](bool from_start, int from) {
        const auto& tree = from_start ? forward : goal_tree;
        auto& queue = from_start ? forward_queue : goal_queue;
        for (const int to : adjacency[static_cast<size_t>(from)]) {
            enqueueEdge(tree, queue, from_start, from, to);
        }
    };

    const auto neighborCount = [&]() {
        const int n = static_cast<int>(vertices.size());
        const double required = std::max(1.0, params_.mire_biait.rgg_factor) *
            std::exp(1.0) * (1.0 + 1.0 / static_cast<double>(NUM_JOINTS)) *
            std::log(std::max(2, n));
        return std::min(n - 1, std::max(1, static_cast<int>(std::ceil(required))));
    };
    const auto linkVertices = [&](int a, int b) {
        if (a == b || std::find(adjacency[static_cast<size_t>(a)].begin(),
                                adjacency[static_cast<size_t>(a)].end(), b) !=
                          adjacency[static_cast<size_t>(a)].end()) {
            return;
        }
        adjacency[static_cast<size_t>(a)].push_back(b);
        adjacency[static_cast<size_t>(b)].push_back(a);
        ++result.effort_stats.graph_edges;
        if (post_phase_started) return;
        updateReverse(a);
        updateReverse(b);
        if (std::isfinite(forward[static_cast<size_t>(a)].cost)) enqueueEdge(forward, forward_queue, true, a, b);
        if (std::isfinite(forward[static_cast<size_t>(b)].cost)) enqueueEdge(forward, forward_queue, true, b, a);
        if (std::isfinite(goal_tree[static_cast<size_t>(a)].cost)) enqueueEdge(goal_tree, goal_queue, false, a, b);
        if (std::isfinite(goal_tree[static_cast<size_t>(b)].cost)) enqueueEdge(goal_tree, goal_queue, false, b, a);
    };
    const auto addVertex = [&](const JointConfig& q, SampleOrigin origin) {
        const int index = static_cast<int>(vertices.size());
        vertices.push_back(q);
        tcp_positions.push_back(fk_.fkine(q, request.tool_model).block<3, 1>(0, 3));
        adjacency.emplace_back();
        vertex_origins.push_back(origin);
        reverse.emplace_back();
        forward.emplace_back();
        goal_tree.emplace_back();
        const int k = neighborCount();
        std::vector<std::pair<double, int>> nearest;
        nearest.reserve(static_cast<size_t>(index));
        for (int other = 0; other < index; ++other) {
            nearest.emplace_back(distance(q, vertices[static_cast<size_t>(other)]), other);
        }
        const int count = std::min(k, static_cast<int>(nearest.size()));
        std::partial_sort(nearest.begin(), nearest.begin() + count, nearest.end());
        for (int rank = 0; rank < count; ++rank) {
            linkVertices(index, nearest[static_cast<size_t>(rank)].second);
        }
        return index;
    };
    const auto refreshTerminalNeighbors = [&]() {
        const int first_sample = static_cast<int>(goals.size()) + 1;
        if (static_cast<int>(vertices.size()) <= first_sample) return;
        const int k = neighborCount();
        std::vector<int> terminals = roots;
        terminals.push_back(start_index);
        for (const int terminal : terminals) {
            std::vector<std::pair<double, int>> nearest;
            nearest.reserve(vertices.size() - static_cast<size_t>(first_sample));
            for (int sample = first_sample; sample < static_cast<int>(vertices.size()); ++sample) {
                nearest.emplace_back(distance(vertices[static_cast<size_t>(terminal)],
                                              vertices[static_cast<size_t>(sample)]), sample);
            }
            const int count = std::min(k, static_cast<int>(nearest.size()));
            if (count == 0) continue;
            std::partial_sort(nearest.begin(), nearest.begin() + count, nearest.end());
            for (int rank = 0; rank < count; ++rank) {
                linkVertices(terminal, nearest[static_cast<size_t>(rank)].second);
            }
        }
    };

    for (const int root : roots) updateReverse(root);

    std::optional<goal_set::DirectPath> direct_incumbent;
    for (const int root : roots) {
        if (!consumeWork()) {
            return finish(false, PlanningFailureCode::kGoalNotReached,
                "MIRE-BiAIT* failed: work budget exhausted during direct connection.");
        }
        if (!validateFullEdge(start_index, root)) continue;
        const double cost = distance(request.q_start, vertices[static_cast<size_t>(root)]);
        if (!direct_incumbent || cost + kEps < direct_incumbent->cost) {
            direct_incumbent = goal_set::DirectPath{
                {request.q_start, vertices[static_cast<size_t>(root)]}, cost, root - 1};
        }
    }
    if (direct_incumbent) {
        best_path = direct_incumbent->path;
        best_cost = direct_incumbent->cost;
        best_tcp_cost = (tcp_positions[static_cast<size_t>(start_index)] -
                         tcp_positions[static_cast<size_t>(direct_incumbent->goal_index + 1)]).norm();
        has_solution = true;
        post_phase_started = true;
        result.effort_stats.selected_goal_root = direct_incumbent->goal_index;
        result.effort_stats.selected_goal_root_before = direct_incumbent->goal_index;
        result.effort_stats.selected_goal_root_after = direct_incumbent->goal_index;
        ++result.effort_stats.root_path_improvements[
            static_cast<size_t>(direct_incumbent->goal_index)];
        result.effort_stats.first_solution_time_s = runtime.elapsedSeconds();
        result.effort_stats.first_solution_path_cost = best_cost;
        result.effort_stats.lower_bound_certified =
            direct_incumbent->reachesLowerBound(goal_lower_bound);
    }

    const auto inInformedUnion = [&](const JointConfig& q) {
        if (!std::isfinite(best_cost)) return true;
        for (const auto& goal : goals) {
            if (distance(request.q_start, q) + distance(q, goal) + kEps < best_cost) return true;
        }
        return false;
    };

    bool sampling_active = true;
    bool post_sampling_active = has_solution;
    int accepted_in_batch = 0;
    int post_batch_attempts = 0;
    std::vector<int> post_root_schedule;
    size_t next_post_root = 0U;
    ++result.effort_stats.batches;
    const auto completeSamplingPhase = [&]() {
        refreshTerminalNeighbors();
        sampling_active = false;
        if (post_sampling_active) {
            post_batch_attempts = 0;
            post_root_schedule.clear();
            next_post_root = 0U;
        }
    };
    const auto sampleAction = [&]() {
        if (!consumeWork()) return false;
        ++result.effort_stats.sample_attempts;
        if (post_sampling_active) {
            ++post_batch_attempts;
            ++result.effort_stats.post_solution_sample_attempts;
        }
        if (post_sampling_active && post_root_schedule.empty()) {
            post_root_schedule = rootSampleSchedule(
                request.q_start, goals, best_cost, post_batch_size);
        }
        SampleOrigin origin = SampleOrigin::kUniform;
        JointConfig q = JointConfig::Zero();
        int sampled_root = -1;
        std::optional<JointConfig> informed;
        if (post_sampling_active && next_post_root < post_root_schedule.size()) {
            sampled_root = post_root_schedule[next_post_root++];
            ++result.effort_stats.informed_sample_attempts;
            ++result.effort_stats.root_sample_attempts[static_cast<size_t>(sampled_root)];
            informed = informed_sampling::singleRoot(
                request.q_start, goals[static_cast<size_t>(sampled_root)],
                best_cost, limits_, rng);
            if (informed) {
                q = *informed;
                origin = SampleOrigin::kInformed;
            }
        } else {
            q = limits_.sampleUniform(rng);
            ++result.effort_stats.uniform_sample_attempts;
        }
        const bool post_batch_complete = post_sampling_active && (
            post_batch_attempts >= post_batch_size ||
            (post_solution_budget > 0 &&
             result.effort_stats.post_solution_sample_attempts >= post_solution_budget));
        const bool missing_sample = sampled_root >= 0 && !informed;
        if (missing_sample || !inInformedUnion(q) || !validState(q)) {
            if (post_batch_complete) completeSamplingPhase();
            return true;
        }
        addVertex(q, origin);
        ++accepted_in_batch;
        ++result.effort_stats.sampled_states;
        if (origin == SampleOrigin::kInformed) {
            ++result.effort_stats.informed_accepted_states;
            ++result.effort_stats.root_accepted_samples[static_cast<size_t>(sampled_root)];
        } else {
            ++result.effort_stats.uniform_accepted_states;
        }
        if ((!post_sampling_active && accepted_in_batch >= batch_size) || post_batch_complete) {
            completeSamplingPhase();
        }
        return true;
    };

    const auto liveEntry = [&](const ForwardEntry& entry,
                               const std::vector<ForwardNode>& tree) {
        if (entry.generation != tree[static_cast<size_t>(entry.from)].generation ||
            edgeState(entry.from, entry.to) == EdgeState::kBlocked) return false;
        const double candidate = tree[static_cast<size_t>(entry.from)].cost +
            distance(vertices[static_cast<size_t>(entry.from)], vertices[static_cast<size_t>(entry.to)]);
        return candidate + kEps < tree[static_cast<size_t>(entry.to)].cost &&
            (!std::isfinite(best_cost) || entry.lower_bound + kEps < best_cost);
    };
    const auto queueLowerBound = [&](bool from_start) {
        const auto& tree = from_start ? forward : goal_tree;
        auto& queue = from_start ? forward_queue : goal_queue;
        while (!queue.empty() && !liveEntry(queue.top(), tree)) queue.pop();
        return queue.empty() ? std::numeric_limits<double>::infinity()
                             : queue.top().lower_bound;
    };
    const auto chooseForward = [&](bool from_start) -> std::optional<ForwardEntry> {
        const auto& tree = from_start ? forward : goal_tree;
        auto& queue = from_start ? forward_queue : goal_queue;
        while (!queue.empty() && !liveEntry(queue.top(), tree)) queue.pop();
        if (queue.empty()) return std::nullopt;
        if (!params_.mire_biait.enable_effort_focal_queue) {
            const ForwardEntry selected = queue.top();
            queue.pop();
            return selected;
        }

        const double focal_limit = focal_factor * queue.top().lower_bound + kEps;
        std::vector<ForwardEntry> focal;
        while (!queue.empty()) {
            const ForwardEntry entry = queue.top();
            if (entry.lower_bound > focal_limit) break;
            queue.pop();
            if (liveEntry(entry, tree)) focal.push_back(entry);
        }
        if (focal.empty()) return std::nullopt;

        const auto compareValue = [](double lhs, double rhs) {
            if (lhs == rhs) return 0;
            if (lhs + kEps < rhs) return -1;
            if (rhs + kEps < lhs) return 1;
            return 0;
        };
        size_t chosen = 0U;
        int chosen_effort = kInfiniteEffort;
        double chosen_estimated_cost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < focal.size(); ++i) {
            const ForwardEntry& entry = focal[i];
            ++result.effort_stats.focal_entries_examined;
            const double candidate = tree[static_cast<size_t>(entry.from)].cost +
                distance(vertices[static_cast<size_t>(entry.from)], vertices[static_cast<size_t>(entry.to)]);
            const CostEffort& reverse_label = reverse[static_cast<size_t>(entry.to)].g;
            int effort = edgeEffort(entry.from, entry.to);
            double estimated_cost = candidate +
                distance(vertices[static_cast<size_t>(start_index)],
                         vertices[static_cast<size_t>(entry.to)]);
            if (from_start) {
                if (reverse_label.effort >= kInfiniteEffort || effort >= kInfiniteEffort) {
                    effort = kInfiniteEffort;
                } else {
                    effort = std::min(kInfiniteEffort, effort + reverse_label.effort);
                }
                estimated_cost = candidate + (std::isfinite(reverse_label.cost)
                    ? reverse_label.cost : minGoalDistance(entry.to));
            }
            bool take = i == 0U;
            if (!take) {
                const ForwardEntry& current = focal[chosen];
                const int estimated_order = compareValue(estimated_cost, chosen_estimated_cost);
                const int lower_bound_order = compareValue(entry.lower_bound, current.lower_bound);
                if (has_solution) {
                    take = estimated_order < 0 ||
                        (estimated_order == 0 && lower_bound_order < 0) ||
                        (estimated_order == 0 && lower_bound_order == 0 && effort < chosen_effort);
                } else {
                    take = effort < chosen_effort ||
                        (effort == chosen_effort && estimated_order < 0) ||
                        (effort == chosen_effort && estimated_order == 0 && lower_bound_order < 0);
                }
                if (!take && effort == chosen_effort && estimated_order == 0 &&
                    lower_bound_order == 0) {
                    take = std::tie(entry.from, entry.to, entry.serial) <
                        std::tie(current.from, current.to, current.serial);
                }
            }
            if (take) {
                chosen = i;
                chosen_effort = effort;
                chosen_estimated_cost = estimated_cost;
            }
        }
        const ForwardEntry selected = focal[chosen];
        for (size_t i = 0; i < focal.size(); ++i) {
            if (i != chosen) queue.push(focal[i]);
        }
        return selected;
    };

    const auto rootForPath = [&](const std::vector<int>& indices) {
        if (indices.empty()) return -1;
        const int endpoint = indices.back();
        const auto found = std::find(roots.begin(), roots.end(), endpoint);
        return found == roots.end() ? -1 : *found;
    };
    const auto acceptIncumbentPath = [&](const std::vector<int>& indices) {
        if (indices.size() < 2U) return false;
        const int root = rootForPath(indices);
        if (root < 1 || root > static_cast<int>(goals.size())) return false;

        double candidate_cost = 0.0;
        for (size_t i = 1; i < indices.size(); ++i) {
            candidate_cost += distance(vertices[static_cast<size_t>(indices[i - 1U])],
                                       vertices[static_cast<size_t>(indices[i])]);
        }
        double candidate_tcp_cost = 0.0;
        for (size_t i = 1; i < indices.size(); ++i) {
            candidate_tcp_cost += (tcp_positions[static_cast<size_t>(indices[i - 1U])] -
                                   tcp_positions[static_cast<size_t>(indices[i])]).norm();
        }
        const bool shorter_joint = candidate_cost + kEps < best_cost;
        const bool equal_joint_shorter_tcp = std::abs(candidate_cost - best_cost) <= kEps &&
            candidate_tcp_cost + kEps < best_tcp_cost;
        if (!shorter_joint && !equal_joint_shorter_tcp) return false;
        for (size_t i = 1; i < indices.size(); ++i) {
            if (!consumeWork()) return false;
            ++result.effort_stats.final_validation_calls;
            const int from = indices[i - 1U];
            const int to = indices[i];
            if (!validState(vertices[static_cast<size_t>(to)]) ||
                !collision_->isMotionValid(vertices[static_cast<size_t>(from)],
                                            vertices[static_cast<size_t>(to)],
                                            validation_distance)) {
                blockEdge(from, to);
                return false;
            }
        }

        std::vector<JointConfig> path;
        path.reserve(indices.size());
        for (const int index : indices) path.push_back(vertices[static_cast<size_t>(index)]);
        const double previous_cost = best_cost;
        const double previous_tcp_cost = best_tcp_cost;
        best_cost = candidate_cost;
        best_tcp_cost = candidate_tcp_cost;
        best_path = std::move(path);
        ++result.effort_stats.incumbent_improvements;
        ++result.effort_stats.root_path_improvements[static_cast<size_t>(root - 1)];
        for (const int index : indices) {
            if (vertex_origins[static_cast<size_t>(index)] == SampleOrigin::kInformed) {
                ++result.effort_stats.informed_path_improvements;
                break;
            }
        }
        result.effort_stats.selected_goal_root = root - 1;
        result.effort_stats.selected_goal_root_after = root - 1;
        if (!has_solution) {
            has_solution = true;
            result.effort_stats.selected_goal_root_before = root - 1;
            result.effort_stats.first_solution_time_s = runtime.elapsedSeconds();
            result.effort_stats.first_solution_path_cost = best_cost;
        } else if (post_phase_started) {
            ++result.effort_stats.post_strict_improvements;
            result.effort_stats.joint_cost_reduction_rad += previous_cost - best_cost;
            result.effort_stats.tcp_cost_reduction_m += previous_tcp_cost - best_tcp_cost;
        }
        if (std::abs(best_cost - goal_lower_bound) <= kEps) {
            result.effort_stats.lower_bound_certified = true;
        }
        return true;
    };

    const auto strictCandidate = [&](int start_node, int goal_node) {
        if (start_node != goal_node && edgeState(start_node, goal_node) != EdgeState::kFree) {
            return false;
        }
        const double bridge_cost = start_node == goal_node ? 0.0 :
            distance(vertices[static_cast<size_t>(start_node)],
                     vertices[static_cast<size_t>(goal_node)]);
        const double tree_cost = forward[static_cast<size_t>(start_node)].cost + bridge_cost +
            goal_tree[static_cast<size_t>(goal_node)].cost;
        const int root = goal_tree[static_cast<size_t>(goal_node)].root;
        if (root < 1 || root > static_cast<int>(goals.size()) || tree_cost + kEps >= best_cost) {
            return false;
        }

        std::vector<int> indices;
        int node = start_node;
        for (size_t guard = 0; node >= 0 && guard <= vertices.size(); ++guard) {
            indices.push_back(node);
            if (node == start_index) break;
            node = forward[static_cast<size_t>(node)].parent;
        }
        if (indices.empty() || indices.back() != start_index) return false;
        std::reverse(indices.begin(), indices.end());
        if (start_node != goal_node) indices.push_back(goal_node);
        node = goal_tree[static_cast<size_t>(goal_node)].parent;
        for (size_t guard = 0; node >= 0 && guard <= vertices.size(); ++guard) {
            indices.push_back(node);
            if (node == root) break;
            node = goal_tree[static_cast<size_t>(node)].parent;
        }
        if (indices.size() < 2U || indices.back() != root) return false;

        return acceptIncumbentPath(indices);
    };

    const auto postSparseCheck = [&](int a, int b) {
        EdgeInfo& info = edges[edgeKey(a, b)];
        if (info.state == EdgeState::kBlocked) return false;
        const auto reject = [&]() {
            ++result.effort_stats.sparse_rejected_edges;
            ++result.effort_stats.full_invalid_edges_avoided;
            blockEdge(a, b);
            return false;
        };
        const JointConfig& from = vertices[static_cast<size_t>(a)];
        const JointConfig& to = vertices[static_cast<size_t>(b)];
        if (!info.midpoint_checked) {
            if (!consumeWork()) return false;
            info.midpoint_checked = true;
            ++result.effort_stats.sparse_state_checks;
            if (!validState(0.5 * (from + to))) return reject();
        }
        if (!info.quarters_checked && distance(from, to) > 2.0 * validation_distance) {
            info.quarters_checked = true;
            for (const double ratio : {0.25, 0.75}) {
                if (!consumeWork()) return false;
                ++result.effort_stats.sparse_state_checks;
                ++result.effort_stats.quarter_state_checks;
                if (!validState(from + ratio * (to - from))) return reject();
            }
        }
        return true;
    };

    const auto postOptimisticPath = [&]() -> std::optional<std::vector<int>> {
        const size_t count = vertices.size();
        std::vector<double> joint_cost(count, std::numeric_limits<double>::infinity());
        std::vector<double> tcp_cost(count, std::numeric_limits<double>::infinity());
        std::vector<int> parent(count, -1);
        std::priority_queue<PostSearchEntry, std::vector<PostSearchEntry>, PostSearchCompare> queue;
        std::uint64_t serial = 0;
        const auto tcpGoalLowerBound = [&](int vertex) {
            double lower = std::numeric_limits<double>::infinity();
            for (const int root : roots) {
                lower = std::min(lower,
                    (tcp_positions[static_cast<size_t>(vertex)] -
                     tcp_positions[static_cast<size_t>(root)]).norm());
            }
            return lower;
        };
        const auto push = [&](int vertex) {
            queue.push({vertex,
                joint_cost[static_cast<size_t>(vertex)] + minGoalDistance(vertex),
                tcp_cost[static_cast<size_t>(vertex)] + tcpGoalLowerBound(vertex), serial++});
        };
        joint_cost[static_cast<size_t>(start_index)] = 0.0;
        tcp_cost[static_cast<size_t>(start_index)] = 0.0;
        push(start_index);

        while (!queue.empty()) {
            while (!queue.empty()) {
                const auto& top = queue.top();
                const double expected = joint_cost[static_cast<size_t>(top.vertex)] +
                    minGoalDistance(top.vertex);
                if (std::abs(top.joint_lower_bound - expected) <= kEps) break;
                queue.pop();
            }
            if (queue.empty() || queue.top().joint_lower_bound > best_cost + kEps) break;

            const double focal_limit = queue.top().joint_lower_bound * 1.01 + kEps;
            std::vector<PostSearchEntry> focal;
            while (!queue.empty() && queue.top().joint_lower_bound <= focal_limit) {
                PostSearchEntry entry = queue.top();
                queue.pop();
                const double expected = joint_cost[static_cast<size_t>(entry.vertex)] +
                    minGoalDistance(entry.vertex);
                if (std::abs(entry.joint_lower_bound - expected) <= kEps) focal.push_back(entry);
            }
            if (focal.empty()) continue;
            const auto chosen = std::min_element(focal.begin(), focal.end(),
                [](const PostSearchEntry& lhs, const PostSearchEntry& rhs) {
                    if (std::abs(lhs.tcp_lower_bound - rhs.tcp_lower_bound) > kEps) {
                        return lhs.tcp_lower_bound < rhs.tcp_lower_bound;
                    }
                    if (std::abs(lhs.joint_lower_bound - rhs.joint_lower_bound) > kEps) {
                        return lhs.joint_lower_bound < rhs.joint_lower_bound;
                    }
                    return lhs.serial < rhs.serial;
                });
            const PostSearchEntry current = *chosen;
            for (const auto& entry : focal) {
                if (entry.serial != current.serial) queue.push(entry);
            }
            if (!consumeWork()) return std::nullopt;
            ++result.effort_stats.post_cost_search_pops;

            if (isRoot(current.vertex)) {
                const double root_joint_cost = joint_cost[static_cast<size_t>(current.vertex)];
                const double root_tcp_cost = tcp_cost[static_cast<size_t>(current.vertex)];
                if (root_joint_cost + kEps >= best_cost &&
                    !(std::abs(root_joint_cost - best_cost) <= kEps &&
                      root_tcp_cost + kEps < best_tcp_cost)) {
                    continue;
                }
                std::vector<int> path;
                for (int vertex = current.vertex; vertex >= 0;
                     vertex = parent[static_cast<size_t>(vertex)]) {
                    path.push_back(vertex);
                    if (vertex == start_index) break;
                }
                if (path.back() != start_index) return std::nullopt;
                std::reverse(path.begin(), path.end());
                return path;
            }
            for (const int next : adjacency[static_cast<size_t>(current.vertex)]) {
                if (edgeState(current.vertex, next) == EdgeState::kBlocked) continue;
                const double next_joint = joint_cost[static_cast<size_t>(current.vertex)] +
                    distance(vertices[static_cast<size_t>(current.vertex)],
                             vertices[static_cast<size_t>(next)]);
                const double next_tcp = tcp_cost[static_cast<size_t>(current.vertex)] +
                    (tcp_positions[static_cast<size_t>(current.vertex)] -
                     tcp_positions[static_cast<size_t>(next)]).norm();
                const bool better_joint = next_joint + kEps < joint_cost[static_cast<size_t>(next)];
                const bool equal_joint_better_tcp =
                    std::abs(next_joint - joint_cost[static_cast<size_t>(next)]) <= kEps &&
                    next_tcp + kEps < tcp_cost[static_cast<size_t>(next)];
                if (!better_joint && !equal_joint_better_tcp) continue;
                joint_cost[static_cast<size_t>(next)] = next_joint;
                tcp_cost[static_cast<size_t>(next)] = next_tcp;
                parent[static_cast<size_t>(next)] = current.vertex;
                push(next);
            }
        }
        return std::nullopt;
    };

    const auto repairPostSolution = [&]() {
        while (result.effort_stats.work_units < max_work && !runtime.shouldStop()) {
            const auto candidate = postOptimisticPath();
            if (!candidate) return;
            ++result.effort_stats.post_candidate_paths;

            bool rejected = false;
            std::vector<std::pair<int, int>> unknown_edges;
            for (size_t i = 1; i < candidate->size(); ++i) {
                const int from = (*candidate)[i - 1U];
                const int to = (*candidate)[i];
                if (edgeState(from, to) == EdgeState::kUnknown) {
                    if (!postSparseCheck(from, to)) {
                        rejected = true;
                        break;
                    }
                    unknown_edges.emplace_back(from, to);
                }
            }
            if (result.effort_stats.work_units >= max_work) return;
            if (rejected) continue;

            std::sort(unknown_edges.begin(), unknown_edges.end(), [&](const auto& lhs, const auto& rhs) {
                return distance(vertices[static_cast<size_t>(lhs.first)],
                                vertices[static_cast<size_t>(lhs.second)]) >
                       distance(vertices[static_cast<size_t>(rhs.first)],
                                vertices[static_cast<size_t>(rhs.second)]);
            });
            for (const auto& edge : unknown_edges) {
                if (!consumeWork()) return;
                if (!validateFullEdge(edge.first, edge.second)) {
                    rejected = true;
                    break;
                }
            }
            if (rejected) continue;
            if (!acceptIncumbentPath(*candidate)) return;
        }
    };

    bool expand_start_on_tie = true;
    while (result.effort_stats.work_units < max_work && !runtime.shouldStop()) {
        runtime.capture(result, has_solution, best_cost, static_cast<int>(vertices.size()));
        if (post_solution_budget > 0 && has_solution &&
            result.effort_stats.post_solution_sample_attempts >= post_solution_budget) {
            if (!sampling_active) repairPostSolution();
            result.effort_stats.post_solution_budget_complete = true;
            break;
        }
        if (sampling_active) {
            if (!sampleAction()) break;
            continue;
        }
        if (post_phase_started) {
            repairPostSolution();
            if (result.effort_stats.work_units >= max_work || runtime.shouldStop()) {
                break;
            }
            sampling_active = true;
            accepted_in_batch = 0;
            post_sampling_active = true;
            ++result.effort_stats.batches;
            continue;
        }
        processReverse();

        const double start_lower_bound = queueLowerBound(true);
        const double goal_lower_bound_value = queueLowerBound(false);
        bool from_start = start_lower_bound < goal_lower_bound_value;
        if (std::abs(start_lower_bound - goal_lower_bound_value) <= kEps) {
            from_start = expand_start_on_tie;
            expand_start_on_tie = !expand_start_on_tie;
        }
        const auto selected = std::isfinite(std::min(start_lower_bound, goal_lower_bound_value))
            ? chooseForward(from_start) : std::optional<ForwardEntry>{};
        if (selected) {
            const ForwardEntry entry = *selected;
            if (!consumeWork()) break;
            if (from_start) {
                ++result.effort_stats.forward_edge_pops;
            } else {
                ++result.effort_stats.goal_tree_edge_pops;
            }
            if (!sparseCheck(entry.from, entry.to)) continue;
            if (!validateFullEdge(entry.from, entry.to)) continue;
            auto& tree = from_start ? forward : goal_tree;
            const double candidate = tree[static_cast<size_t>(entry.from)].cost +
                distance(vertices[static_cast<size_t>(entry.from)], vertices[static_cast<size_t>(entry.to)]);
            if (candidate + kEps >= tree[static_cast<size_t>(entry.to)].cost) continue;
            ForwardNode& target = tree[static_cast<size_t>(entry.to)];
            target.cost = candidate;
            target.parent = entry.from;
            target.root = from_start ? -1 : tree[static_cast<size_t>(entry.from)].root;
            ++target.generation;
            enqueueFrom(from_start, entry.to);
            const auto& opposite = from_start ? goal_tree : forward;
            bool improved = std::isfinite(opposite[static_cast<size_t>(entry.to)].cost) &&
                strictCandidate(entry.to, entry.to);
            for (const int neighbor : adjacency[static_cast<size_t>(entry.to)]) {
                if (edgeState(entry.to, neighbor) != EdgeState::kFree) continue;
                if (from_start && std::isfinite(goal_tree[static_cast<size_t>(neighbor)].cost)) {
                    improved = strictCandidate(entry.to, neighbor) || improved;
                } else if (!from_start &&
                           std::isfinite(forward[static_cast<size_t>(neighbor)].cost)) {
                    improved = strictCandidate(neighbor, entry.to) || improved;
                }
            }
            if (improved) {
                if (!post_phase_started) {
                    post_phase_started = true;
                    post_sampling_active = true;
                    sampling_active = true;
                    accepted_in_batch = 0;
                    post_batch_attempts = 0;
                    post_root_schedule.clear();
                    next_post_root = 0U;
                    ++result.effort_stats.batches;
                }
            }
            continue;
        }

        if (!reverse_queue.empty()) continue;
        if (runtime.shouldStop()) break;
        sampling_active = true;
        accepted_in_batch = 0;
        post_sampling_active = has_solution;
        ++result.effort_stats.batches;
    }

    result.num_nodes = static_cast<int>(vertices.size());
    std::ostringstream diagnostics;
    diagnostics << "MIRE-BiAIT*: seed=" << effective_seed
                << " batches=" << result.effort_stats.batches
                << " work=" << result.effort_stats.work_units
                << " uniform=" << result.effort_stats.uniform_sample_attempts
                << " informed=" << result.effort_stats.informed_sample_attempts
                << " uniform_accepted=" << result.effort_stats.uniform_accepted_states
                << " informed_accepted=" << result.effort_stats.informed_accepted_states
                << " reverse_pops=" << result.effort_stats.reverse_queue_pops
                << " reverse_stale=" << result.effort_stats.reverse_queue_stale_discards
                << " reverse_consistent=" << result.effort_stats.reverse_queue_consistent_discards
                << " forward_pops=" << result.effort_stats.forward_edge_pops
                << " goal_pops=" << result.effort_stats.goal_tree_edge_pops
                << " focal_examined=" << result.effort_stats.focal_entries_examined
                << " sparse_checks=" << result.effort_stats.sparse_state_checks
                << " edge_checks=" << result.effort_stats.edge_validation_attempts
                << " blocked=" << result.effort_stats.blocked_edges
                << " roots=" << result.effort_stats.goal_root_count
                << " root=" << result.effort_stats.selected_goal_root
                << " post_pops=" << result.effort_stats.post_cost_search_pops
                << " post_candidates=" << result.effort_stats.post_candidate_paths
                << " post_improvements=" << result.effort_stats.post_strict_improvements
                << " post_samples=" << result.effort_stats.post_solution_sample_attempts
                << " post_budget_complete="
                << (result.effort_stats.post_solution_budget_complete ? "true" : "false")
                << " sparse_rejected=" << result.effort_stats.sparse_rejected_edges
                << " lower_bound=" << (result.effort_stats.lower_bound_certified ? "true" : "false")
                << " budget_exhausted=" << (result.effort_stats.budget_exhausted ? "true" : "false");
    result.diagnostics = diagnostics.str();
    if (best_path.empty()) {
        return finish(false, runtime.deadlineReached() ? PlanningFailureCode::kTimeout : PlanningFailureCode::kGoalNotReached,
            runtime.deadlineReached() ? "deadline_no_solution" : "MIRE-BiAIT* failed: no strictly valid path found.");
    }
    result.path = std::move(best_path);
    result.path_cost = best_cost;
    return finish(true, PlanningFailureCode::kNone, "");
}

PlanResult MireBiAitStar::plan(
    const JointConfig& q_start, const JointConfig& q_goal,
    const Vector3d& p_start, const Vector3d& p_goal,
    const RotMatrix3d& R_target, const Vector3d& obs_origin,
    const Vector3d& obs_size) {
    (void)p_start;
    (void)p_goal;
    (void)R_target;
    (void)obs_origin;
    (void)obs_size;
    PlanRequestCore request;
    request.q_start = q_start;
    request.q_goal = q_goal;
    return plan(request);
}

}  // namespace fairino_planning
