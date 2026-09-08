#include <gtest/gtest.h>

#include "myrobot_planning_core/algorithms/aapf_bi_rrt_star.h"
#include "myrobot_planning_core/aapf/aapf_adaptive_sampler_selector.h"
#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/dh_kinematics.h"

#include <array>
#include <chrono>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace fairino_planning {
namespace {

class ClearanceCapableCollision : public CollisionInterface {
public:
    bool supportsWorldClearance() const override { return true; }
    ClearanceQueryResult nearestWorldClearance(const JointConfig&, double) const override {
        ClearanceQueryResult result;
        result.supported = true;
        return result;
    }
};

class AlwaysValidCollision final : public ClearanceCapableCollision {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        return true;
    }
};

class AlwaysInvalidCollision final : public ClearanceCapableCollision {
public:
    bool isStateValid(const JointConfig&) const override { return false; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls;
        return false;
    }

    mutable int motion_calls = 0;
};

class RejectAllMotionCollision final : public ClearanceCapableCollision {
public:
    RejectAllMotionCollision(JointConfig q_start, JointConfig q_goal)
        : q_start_(std::move(q_start)), q_goal_(std::move(q_goal)) {}

    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& from, const JointConfig& to, double) const override {
        ++motion_calls;
        saw_direct_edge = saw_direct_edge ||
            ((from - q_start_).norm() < 1e-12 && (to - q_goal_).norm() < 1e-12) ||
            ((from - q_goal_).norm() < 1e-12 && (to - q_start_).norm() < 1e-12);
        return false;
    }

    mutable bool saw_direct_edge = false;
    mutable int motion_calls = 0;

private:
    JointConfig q_start_;
    JointConfig q_goal_;
};

class DirectStartGoalEdgeInvalidCollision final : public ClearanceCapableCollision {
public:
    DirectStartGoalEdgeInvalidCollision(JointConfig q_start, JointConfig q_goal)
        : q_start_(std::move(q_start)), q_goal_(std::move(q_goal)) {}

    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& from, const JointConfig& to, double) const override {
        const bool direct =
            ((from - q_start_).norm() < 1e-12 && (to - q_goal_).norm() < 1e-12) ||
            ((from - q_goal_).norm() < 1e-12 && (to - q_start_).norm() < 1e-12);
        return !direct;
    }

private:
    JointConfig q_start_;
    JointConfig q_goal_;
};

class StartInvalidCollision final : public ClearanceCapableCollision {
public:
    explicit StartInvalidCollision(JointConfig q_start) : q_start_(std::move(q_start)) {}

    bool isStateValid(const JointConfig& q) const override {
        return (q - q_start_).norm() >= 1e-12;
    }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls;
        return true;
    }

    mutable int motion_calls = 0;

private:
    JointConfig q_start_;
};

class RecordingCollision final : public ClearanceCapableCollision {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double distance) const override {
        motion_distances.push_back(distance);
        return true;
    }

    mutable std::vector<double> motion_distances;
};

class UnsupportedClearanceCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        return true;
    }
};

PlanRequestCore exactGoalRequest(const JointConfig& q_start, const JointConfig& q_goal) {
    DHKinematics fk;
    const Transform4d start_pose = fk.fkine(q_start, ToolModel::FLANGE);
    const Transform4d goal_pose = fk.fkine(q_goal, ToolModel::FLANGE);
    PlanRequestCore request;
    request.q_start = q_start;
    request.q_goal = q_goal;
    request.p_start = start_pose.block<3, 1>(0, 3);
    request.p_goal = goal_pose.block<3, 1>(0, 3);
    request.R_target = goal_pose.block<3, 3>(0, 0);
    request.goal_candidates = {q_goal};
    request.started = std::chrono::steady_clock::now();
    request.deadline = request.started + std::chrono::milliseconds(100);
    return request;
}

void observe(
    AapfAdaptiveSamplerSelector& selector,
    AapfSampleSource source,
    double utility,
    double elapsed_ms) {
    selector.observe(source, utility, elapsed_ms);
}

TEST(AapfAdaptiveSamplerSelectorTest, WarmupIsBalancedAndEveryEighthPullIsGlobal) {
    AapfAdaptiveSamplerSelector selector(32, 0.7, 8);
    const std::array<AapfSampleSource, 7> expected{
        AapfSampleSource::kGuided, AapfSampleSource::kGlobal,
        AapfSampleSource::kGuided, AapfSampleSource::kGlobal,
        AapfSampleSource::kGuided, AapfSampleSource::kGlobal,
        AapfSampleSource::kGuided,
    };
    for (const auto source : expected) {
        EXPECT_EQ(selector.choose(), source);
        observe(selector, source, 0.0, 1.0);
    }
    EXPECT_EQ(selector.choose(), AapfSampleSource::kGlobal);
}

TEST(AapfAdaptiveSamplerSelectorTest, PrefersUsefulLowCostSource) {
    AapfAdaptiveSamplerSelector selector(32, 0.7, 0);
    for (int i = 0; i < 2; ++i) {
        observe(selector, AapfSampleSource::kGuided, 1.0, 10.0);
        observe(selector, AapfSampleSource::kGlobal, 1.0, 1.0);
    }
    EXPECT_EQ(selector.choose(), AapfSampleSource::kGlobal);

    AapfAdaptiveSamplerSelector guided_selector(32, 0.7, 0);
    for (int i = 0; i < 2; ++i) {
        observe(guided_selector, AapfSampleSource::kGuided, 1.0, 1.0);
        observe(guided_selector, AapfSampleSource::kGlobal, 0.0, 10.0);
    }
    EXPECT_EQ(guided_selector.choose(), AapfSampleSource::kGuided);
}

TEST(AapfAdaptiveSamplerSelectorTest, TreesDoNotShareHistory) {
    AapfAdaptiveSamplerSelector tree_a(32, 0.7, 0);
    AapfAdaptiveSamplerSelector tree_b(32, 0.7, 0);
    for (int i = 0; i < 2; ++i) {
        observe(tree_a, AapfSampleSource::kGuided, 1.0, 1.0);
        observe(tree_a, AapfSampleSource::kGlobal, 0.0, 1.0);
        observe(tree_b, AapfSampleSource::kGuided, 0.0, 1.0);
        observe(tree_b, AapfSampleSource::kGlobal, 1.0, 1.0);
    }
    EXPECT_EQ(tree_a.choose(), AapfSampleSource::kGuided);
    EXPECT_EQ(tree_b.choose(), AapfSampleSource::kGlobal);
}

TEST(AapfBiRRTStarTest, ExactGoalDirectPathPreservesEndpoint) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kNone);
    ASSERT_EQ(result.path.size(), 2U);
    EXPECT_NEAR((result.path.front() - q_start).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result.path.back() - q_goal).norm(), 0.0, 1e-12);
    EXPECT_NEAR(result.path_cost, 0.1, 1e-12);
}

TEST(AapfBiRRTStarTest, FixedPostSolutionSamplingBudgetStopsAfterExactAttemptCount) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    auto request = exactGoalRequest(q_start, q_goal);

    PlanningParams params;
    params.max_iterations = 20;
    params.post_solution_sample_attempts = 3;
    params.aapf.enable = false;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.sample_attempts, 3);
    EXPECT_EQ(result.effort_stats.post_solution_sample_attempts, 3);
    EXPECT_TRUE(result.effort_stats.post_solution_budget_complete);
}

TEST(AapfBiRRTStarTest, EnabledGuidanceRequiresClearanceCapability) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<UnsupportedClearanceCollision>());
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kInvalidInput);
    EXPECT_NE(result.message.find("clearance"), std::string::npos);
}

TEST(AapfBiRRTStarTest, GlobalOnlyAblationDoesNotRequireClearanceCapability) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    PlanningParams params;
    params.aapf.enable = false;

    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<UnsupportedClearanceCollision>());
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_TRUE(result.success) << result.message;
}

TEST(AapfBiRRTStarTest, MultiRootDirectPathUsesShortestValidCandidate) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_preferred = JointConfig::Zero();
    JointConfig q_alternate = JointConfig::Zero();
    q_preferred[0] = 0.2;
    q_alternate[0] = 0.1;

    auto request = exactGoalRequest(q_start, q_preferred);
    request.goal_candidates = {q_preferred, q_alternate};

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_NEAR((result.path.back() - q_alternate).norm(), 0.0, 1e-12);
    EXPECT_NEAR(result.path_cost, 0.1, 1e-12);
}

TEST(AapfBiRRTStarTest, GoalStateCollisionFailsBeforeSearch) {
    AapfBiRRTStar planner;
    const auto collision = std::make_shared<AlwaysInvalidCollision>();
    planner.setCollisionChecker(collision);

    const PlanResult result = planner.plan(PlanRequestCore{});

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_NE(result.message.find("requested goal joint target is invalid or in collision"),
              std::string::npos);
    EXPECT_EQ(collision->motion_calls, 0);
}

TEST(AapfBiRRTStarTest, ExactGoalRejectsInvalidDirectMotion) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<RejectAllMotionCollision>(q_start, q_goal);

    PlanningParams params;
    params.max_iterations = 0;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(collision);
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_TRUE(collision->saw_direct_edge);
}

TEST(AapfBiRRTStarTest, ExactGoalSearchStillUsesAapfGuidance) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.45;

    PlanningParams params;
    params.max_iterations = 500;
    params.max_step = 0.12;
    params.connect_max_steps = 20;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(
        std::make_shared<DirectStartGoalEdgeInvalidCollision>(q_start, q_goal));

    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.random_seed = 17;
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    ASSERT_GE(result.path.size(), 2U);
    EXPECT_NEAR((result.path.front() - q_start).norm(), 0.0, 1e-12);
    EXPECT_NEAR((result.path.back() - q_goal).norm(), 0.0, 1e-12);
}

TEST(AapfBiRRTStarTest, ContinuesSearchWithinSafetyGuard) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.45;

    PlanningParams params;
    params.max_iterations = 500;
    params.max_step = 0.12;
    params.connect_max_steps = 20;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(
        std::make_shared<DirectStartGoalEdgeInvalidCollision>(q_start, q_goal));

    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.random_seed = 17;
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_LE(result.iterations, params.max_iterations);
}

TEST(AapfBiRRTStarTest, SingleRequestIterationBudgetDoesNotRestartSearch) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.45;
    const auto collision = std::make_shared<RejectAllMotionCollision>(q_start, q_goal);

    PlanningParams params;
    params.max_iterations = 1;
    params.aapf.shrink_motion_attempts = 0;
    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(collision);

    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.iterations, 1) << result.message << " " << result.diagnostics;
    EXPECT_LE(collision->motion_calls, 2);
    ASSERT_FALSE(result.sampling_stats.empty());
    for (const auto& stats : result.sampling_stats) {
        EXPECT_EQ(stats.insertions, 0);
        EXPECT_EQ(stats.progress_events, 0);
        EXPECT_DOUBLE_EQ(stats.utility_sum, 0.0);
    }
}

TEST(AapfBiRRTStarTest, RejectsNonFiniteStartBeforeCollisionQueries) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.q_start[0] = std::numeric_limits<double>::quiet_NaN();

    AapfBiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kInvalidInput);
}

TEST(AapfBiRRTStarTest, StartCollisionFailsBeforeSearch) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<StartInvalidCollision>(q_start);

    AapfBiRRTStar planner;
    planner.setCollisionChecker(collision);
    const PlanResult result = planner.plan(exactGoalRequest(q_start, q_goal));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kCollision);
    EXPECT_EQ(collision->motion_calls, 0);
}

TEST(AapfBiRRTStarTest, UsesPlannerValidationDistanceDuringSearch) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    const auto collision = std::make_shared<RecordingCollision>();
    PlanningParams params;
    params.max_iterations = 1;
    params.validation_distance = 0.1;
    params.aapf.strict_validation_distance = 0.01;

    AapfBiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(collision);
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.random_seed = 31;
    planner.plan(request);

    ASSERT_FALSE(collision->motion_distances.empty());
    for (double distance : collision->motion_distances) {
        EXPECT_LE(distance, 0.1 + 1e-12);
    }
}

TEST(AapfBiRRTStarTest, RequestSeedReproducesSearchPath) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    PlanningParams params;
    params.max_iterations = 1;
    PlanRequestCore request = exactGoalRequest(q_start, q_goal);
    request.random_seed = 97;

    AapfBiRRTStar first;
    first.setParams(params);
    first.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult first_result = first.plan(request);

    AapfBiRRTStar second;
    second.setParams(params);
    second.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult second_result = second.plan(request);

    ASSERT_EQ(first_result.success, second_result.success);
    ASSERT_EQ(first_result.failure_code, second_result.failure_code);
    ASSERT_EQ(first_result.path.size(), second_result.path.size());
    for (size_t i = 0; i < first_result.path.size(); ++i) {
        EXPECT_NEAR((first_result.path[i] - second_result.path[i]).norm(), 0.0, 1e-12);
    }
}

}  // namespace
}  // namespace fairino_planning
