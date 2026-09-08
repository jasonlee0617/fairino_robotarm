#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "myrobot_planning_core/algorithms/mire_bi_ait_star.h"
#include "myrobot_planning_core/collision/collision_interface.h"

namespace fairino_planning {
namespace {

class AlwaysValidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override { return true; }
};

class MiddleBandCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig& q) const override {
        return q[0] < 0.18 || q[0] > 0.22;
    }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double distance) const override {
        const int steps = std::max(2, static_cast<int>(std::ceil((a - b).norm() / distance)) + 1);
        for (int i = 1; i < steps; ++i) {
            if (!isStateValid(a + (static_cast<double>(i) / steps) * (b - a))) return false;
        }
        return true;
    }
};

class DetourEdgeCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        ++motion_calls;
        const bool crosses_middle = std::min(a[0], b[0]) < 0.20 && std::max(a[0], b[0]) > 0.20;
        return !crosses_middle || std::abs(a[1]) > 0.05 || std::abs(b[1]) > 0.05;
    }

    mutable int motion_calls = 0;
};

class BlockAllEdgesCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls;
        return false;
    }

    mutable int motion_calls = 0;
};

class FarRootOnlyDirectCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        const bool start_to_near = std::abs(a[0]) < 1e-12 && std::abs(b[0] - 0.20) < 1e-12;
        return !start_to_near;
    }
};

class FarRootIncumbentCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        const bool from_start = a.norm() < 1e-12;
        return !from_start || b[0] > 0.50;
    }
};

PlanRequestCore requestTo(const JointConfig& goal) {
    PlanRequestCore request;
    request.q_start = JointConfig::Zero();
    request.q_goal = goal;
    request.random_seed = 7;
    return request;
}

PlanningParams params() {
    PlanningParams result;
    result.max_iterations = 128;
    result.validation_distance = 0.03;
    result.mire_biait.batch_size = 32;
    return result;
}

TEST(MireBiAitStarTest, ReturnsStrictlyValidDirectPath) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.10;
    MireBiAitStar planner;
    planner.setParams(params());
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());

    const PlanResult result = planner.plan(requestTo(goal));
    ASSERT_TRUE(result.success) << result.message;
    ASSERT_GE(result.path.size(), 2U);
    EXPECT_NEAR((result.path.front() - JointConfig::Zero()).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result.path.back() - goal).norm(), 0.0, 1e-12);
    EXPECT_GT(result.effort_stats.state_validation_calls, 0);
}

TEST(MireBiAitStarTest, FixedPostSolutionSamplingBudgetStopsAfterExactAttemptCount) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.10;
    auto config = params();
    config.max_iterations = 64;
    config.mire_biait.post_solution_batch_size = 10;
    config.post_solution_sample_attempts = 3;

    MireBiAitStar planner;
    planner.setParams(config);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(requestTo(goal));

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.effort_stats.post_solution_sample_attempts, 3);
    EXPECT_TRUE(result.effort_stats.post_solution_budget_complete);
}

TEST(MireBiAitStarTest, SelectsAnyCollisionFreeGoalRoot) {
    JointConfig preferred = JointConfig::Zero();
    preferred[0] = 0.40;
    JointConfig alternate = JointConfig::Zero();
    alternate[0] = 0.10;
    auto request = requestTo(preferred);
    request.goal_candidates = {preferred, alternate};

    MireBiAitStar planner;
    planner.setParams(params());
    planner.setCollisionChecker(std::make_shared<MiddleBandCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_NEAR((result.path.back() - alternate).norm(), 0.0, 1e-12);
    EXPECT_GT(result.effort_stats.edge_validation_attempts, 0);
}

TEST(MireBiAitStarTest, RespectsValidationBudget) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    MireBiAitStar planner;
    auto limited = params();
    limited.max_iterations = 9;
    limited.mire_biait.batch_size = 3;
    planner.setParams(limited);
    planner.setCollisionChecker(std::make_shared<MiddleBandCollision>());
    const PlanResult result = planner.plan(requestTo(goal));

    EXPECT_LE(result.iterations, limited.max_iterations);
    EXPECT_EQ(result.iterations, result.effort_stats.work_units);
    EXPECT_LE(result.effort_stats.sample_attempts, limited.max_iterations);
}

TEST(MireBiAitStarTest, BoundsSamplesAndEdgeChecksInOneWorkBudget) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto limited = params();
    limited.max_iterations = 11;
    limited.mire_biait.batch_size = 4;
    auto collision = std::make_shared<BlockAllEdgesCollision>();
    MireBiAitStar planner;
    planner.setParams(limited);
    planner.setCollisionChecker(collision);

    const PlanResult result = planner.plan(requestTo(goal));

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.effort_stats.budget_exhausted);
    EXPECT_EQ(result.iterations, result.effort_stats.work_units);
    EXPECT_LE(result.effort_stats.work_units, limited.max_iterations);
    EXPECT_EQ(collision->motion_calls,
              result.effort_stats.edge_validation_attempts +
                  result.effort_stats.final_validation_calls);
}

TEST(MireBiAitStarTest, UsesIncrementalQueuesAndCachesSearchEdges) {
    JointConfig primary = JointConfig::Zero();
    primary[0] = 0.40;
    JointConfig alternate = JointConfig::Zero();
    alternate[0] = 0.45;
    auto request = requestTo(primary);
    request.goal_candidates = {primary, alternate};

    auto collision = std::make_shared<DetourEdgeCollision>();
    MireBiAitStar planner;
    auto roomy = params();
    roomy.max_iterations = 512;
    planner.setParams(roomy);
    planner.setCollisionChecker(collision);
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message << "\n" << result.diagnostics;
    EXPECT_GT(result.effort_stats.reverse_queue_pops, 0);
    EXPECT_GT(result.effort_stats.forward_edge_pops, 0);
    EXPECT_GT(result.effort_stats.goal_tree_edge_pops, 0);
    EXPECT_GT(result.effort_stats.graph_edges, 0);
    EXPECT_LT(result.effort_stats.edge_validation_attempts,
              result.effort_stats.graph_edges);
    EXPECT_EQ(result.effort_stats.reverse_queue_consistent_discards, 0);
    EXPECT_EQ(collision->motion_calls,
              result.effort_stats.edge_validation_attempts +
                  result.effort_stats.final_validation_calls);
}

TEST(MireBiAitStarTest, RecordsActualSeedAndBoundsAllQueueActions) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto limited = params();
    limited.max_iterations = 12;
    limited.mire_biait.batch_size = 3;
    MireBiAitStar planner;
    planner.setParams(limited);
    planner.setCollisionChecker(std::make_shared<BlockAllEdgesCollision>());
    auto request = requestTo(goal);
    request.random_seed = 29;

    const PlanResult result = planner.plan(request);

    EXPECT_LE(result.effort_stats.work_units, limited.max_iterations);
    EXPECT_NE(result.diagnostics.find("seed=29"), std::string::npos);
    EXPECT_LE(result.effort_stats.reverse_queue_pops + result.effort_stats.forward_edge_pops +
                  result.effort_stats.goal_tree_edge_pops,
              limited.max_iterations);
}

TEST(MireBiAitStarTest, FarDirectRootIsAnIncumbentNotAnEarlyExit) {
    JointConfig near_goal = JointConfig::Zero();
    near_goal[0] = 0.20;
    JointConfig far_goal = JointConfig::Zero();
    far_goal[0] = 0.60;
    auto request = requestTo(near_goal);
    request.goal_candidates = {near_goal, far_goal};
    auto planning_params = params();
    planning_params.max_iterations = 128;
    MireBiAitStar planner;
    planner.setParams(planning_params);
    planner.setCollisionChecker(std::make_shared<FarRootOnlyDirectCollision>());

    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GE(result.iterations, 7);
    EXPECT_FALSE(result.effort_stats.lower_bound_certified);
}

TEST(MireBiAitStarTest, UsesRootInformedSamplingAfterFirstSolution) {
    JointConfig near_goal = JointConfig::Zero();
    near_goal[0] = 0.20;
    JointConfig far_goal = JointConfig::Zero();
    far_goal[0] = 0.60;
    auto request = requestTo(near_goal);
    request.goal_candidates = {near_goal, far_goal};
    auto planning_params = params();

    MireBiAitStar planner;
    planner.setParams(planning_params);
    planner.setCollisionChecker(std::make_shared<FarRootOnlyDirectCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.effort_stats.informed_sample_attempts, 0);
    EXPECT_LE(result.effort_stats.informed_sample_attempts, planning_params.max_iterations);
}

TEST(MireBiAitStarTest, AllocatesInformedSamplesAcrossEligibleRootsDeterministically) {
    JointConfig near_goal = JointConfig::Zero();
    near_goal[0] = 0.20;
    JointConfig middle_goal = JointConfig::Zero();
    middle_goal[0] = 0.35;
    JointConfig far_goal = JointConfig::Zero();
    far_goal[0] = 0.60;
    auto request = requestTo(near_goal);
    request.goal_candidates = {near_goal, middle_goal, far_goal};
    auto planning_params = params();
    planning_params.max_iterations = 256;

    const auto run = [&]() {
        MireBiAitStar planner;
        planner.setParams(planning_params);
        planner.setCollisionChecker(std::make_shared<FarRootIncumbentCollision>());
        return planner.plan(request);
    };
    const PlanResult first = run();
    const PlanResult second = run();

    ASSERT_TRUE(first.success) << first.message;
    ASSERT_EQ(first.effort_stats.root_sample_attempts.size(), 3U);
    EXPECT_GT(first.effort_stats.root_sample_attempts[0], 0);
    EXPECT_GT(first.effort_stats.root_sample_attempts[1], 0);
    EXPECT_EQ(first.effort_stats.root_sample_attempts[2], 0);
    EXPECT_GT(first.effort_stats.root_accepted_samples[0], 0);
    EXPECT_GT(first.effort_stats.root_accepted_samples[1], 0);
    EXPECT_EQ(first.effort_stats.root_sample_attempts,
              second.effort_stats.root_sample_attempts);
    EXPECT_GT(first.effort_stats.post_cost_search_pops, 0);
    EXPECT_GT(first.effort_stats.quarter_state_checks, 0);
    EXPECT_EQ(first.effort_stats.reverse_queue_pops, 0);
    ASSERT_EQ(first.path.size(), second.path.size());
    for (size_t i = 0; i < first.path.size(); ++i) {
        EXPECT_NEAR((first.path[i] - second.path[i]).norm(), 0.0, 1e-12);
    }
}

TEST(MireBiAitStarTest, ProcessesInformedSamplesInMicroBatches) {
    JointConfig near_goal = JointConfig::Zero();
    near_goal[0] = 0.20;
    JointConfig far_goal = JointConfig::Zero();
    far_goal[0] = 0.60;
    auto request = requestTo(near_goal);
    request.goal_candidates = {near_goal, far_goal};
    auto planning_params = params();
    planning_params.mire_biait.post_solution_batch_size = 5;
    planning_params.max_iterations = 512;

    MireBiAitStar planner;
    planner.setParams(planning_params);
    planner.setCollisionChecker(std::make_shared<FarRootIncumbentCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GE(result.effort_stats.batches, 3);
}

TEST(MireBiAitStarTest, IncumbentOptimizationUsesCostSearchWithoutReversePropagation) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.10;
    auto planning_params = params();
    planning_params.mire_biait.post_solution_batch_size = 5;
    planning_params.max_iterations = 256;

    MireBiAitStar planner;
    planner.setParams(planning_params);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(requestTo(goal));

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.effort_stats.reverse_queue_pops, 0);
    EXPECT_GT(result.effort_stats.post_cost_search_pops, 0);
    EXPECT_EQ(result.effort_stats.selected_goal_root_before, 0);
    EXPECT_EQ(result.effort_stats.selected_goal_root_after, 0);
}

TEST(MireBiAitStarTest, CountsOnlyLiveReverseQueueWork) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto roomy = params();
    roomy.max_iterations = 128;
    auto collision = std::make_shared<DetourEdgeCollision>();
    MireBiAitStar planner;
    planner.setParams(roomy);
    planner.setCollisionChecker(collision);

    const PlanResult result = planner.plan(requestTo(goal));

    ASSERT_TRUE(result.success) << result.message << "\n" << result.diagnostics;
    EXPECT_LE(result.effort_stats.reverse_queue_pops, result.effort_stats.work_units);
    EXPECT_EQ(result.effort_stats.reverse_queue_consistent_discards, 0);
    EXPECT_EQ(collision->motion_calls,
              result.effort_stats.edge_validation_attempts +
                  result.effort_stats.final_validation_calls);
}

}  // namespace
}  // namespace fairino_planning
