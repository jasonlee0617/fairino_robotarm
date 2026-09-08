#include <gtest/gtest.h>

#include "myrobot_planning_core/algorithms/bi_rrt_star.h"
#include "myrobot_planning_core/algorithms/rrt_star.h"
#include "myrobot_planning_core/collision/collision_interface.h"
#include "myrobot_planning_core/dh_kinematics.h"

#include <cmath>
#include <memory>
#include <vector>

namespace fairino_planning {
namespace {

// ── Controlled collision checkers ──

class AlwaysValidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        return true;
    }
};

class AlwaysInvalidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return false; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        return false;
    }
};

// Only the stored goal config is invalid; everything else passes.
class GoalOnlyInvalidCollision final : public CollisionInterface {
public:
    explicit GoalOnlyInvalidCollision(JointConfig goal) : goal_(std::move(goal)) {}
    bool isStateValid(const JointConfig& q) const override {
        return (q - goal_).norm() >= 1e-12;
    }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_calls_;
        return true;
    }
    int motionCalls() const { return motion_calls_; }
private:
    JointConfig goal_;
    mutable int motion_calls_ = 0;
};

// Short segments valid, long direct line invalid — forces multi-step bridge.
class ShortOnlyCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        return (a - b).norm() < 0.5;
    }
};

// All bridge segments valid, but the final edge to the other tree is rejected.
class BridgeValidFinalEdgeInvalidCollision final : public CollisionInterface {
public:
    explicit BridgeValidFinalEdgeInvalidCollision(JointConfig goal) : goal_(std::move(goal)) {}
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        // Reject any segment that touches the goal config.
        if ((a - goal_).norm() < 1e-10 || (b - goal_).norm() < 1e-10) return false;
        return true;
    }
private:
    JointConfig goal_;
};

// ── Helpers ──

PlanRequestCore makeRequest(const JointConfig& q_start, const JointConfig& q_goal) {
    DHKinematics fk;
    PlanRequestCore req;
    req.q_start = q_start;
    req.q_goal = q_goal;
    Transform4d T_start = fk.fkine(q_start, ToolModel::FLANGE);
    Transform4d T_goal = fk.fkine(q_goal, ToolModel::FLANGE);
    req.p_start = T_start.block<3, 1>(0, 3);
    req.p_goal = T_goal.block<3, 1>(0, 3);
    req.R_target = T_goal.block<3, 3>(0, 0);
    return req;
}

bool endsAtCandidate(const PlanResult& result, const std::vector<JointConfig>& candidates) {
    for (const auto& candidate : candidates) {
        if ((result.path.back() - candidate).norm() < 1e-10) return true;
    }
    return false;
}

// ── Tests ──

TEST(BiRRTStarTest, ExactGoalDirectPathPreservesEndpoint) {
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;

    auto req = makeRequest(q_start, q_goal);

    BiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    PlanResult result = planner.plan(req);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kNone);
    EXPECT_GE(result.path.size(), 2U);
    EXPECT_LT((result.path.front() - q_start).norm(), 1e-10);
    EXPECT_LT((result.path.back() - q_goal).norm(), 1e-10);
}

TEST(BiRRTStarTest, FixedPostSolutionSamplingBudgetStopsAfterExactAttemptCount) {
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.1;
    auto request = makeRequest(JointConfig::Zero(), q_goal);
    PlanningParams params;
    params.max_iterations = 20;
    params.post_solution_sample_attempts = 3;

    BiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.sample_attempts, 3);
    EXPECT_EQ(result.effort_stats.post_solution_sample_attempts, 3);
    EXPECT_TRUE(result.effort_stats.post_solution_budget_complete);
}

TEST(RRTStarTest, DirectPathPreservesEndpoint) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.1;
    RRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());

    const PlanResult result = planner.plan(makeRequest(JointConfig::Zero(), goal));
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_NEAR((result.path.back() - goal).norm(), 0.0, 1e-12);
}

TEST(RRTStarTest, FixedPostSolutionSamplingBudgetStopsAfterExactAttemptCount) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.1;
    auto request = makeRequest(JointConfig::Zero(), goal);
    PlanningParams params;
    params.max_iterations = 20;
    params.post_solution_sample_attempts = 3;

    RRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.sample_attempts, 3);
    EXPECT_EQ(result.effort_stats.post_solution_sample_attempts, 3);
    EXPECT_TRUE(result.effort_stats.post_solution_budget_complete);
}

TEST(SharedGoalCandidatesTest, CustomPlannersCanReachAnyCandidate) {
    const JointConfig q_start = JointConfig::Zero();
    JointConfig q_preferred = JointConfig::Zero();
    JointConfig q_alternate = JointConfig::Zero();
    q_preferred[0] = 0.10;
    q_alternate[0] = 0.05;
    auto request = makeRequest(q_start, q_preferred);
    request.goal_candidates = {q_preferred, q_alternate};

    PlanningParams params;
    params.max_iterations = 16;
    params.max_step = 0.20;
    params.goal_bias = 1.0;

    const auto check_result = [&](const PlanResult& result) {
        ASSERT_TRUE(result.success) << result.message;
        ASSERT_FALSE(result.path.empty());
        EXPECT_TRUE(endsAtCandidate(result, request.goal_candidates));
    };

    BiRRTStar birrt;
    birrt.setParams(params);
    birrt.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    check_result(birrt.plan(request));

    RRTStar rrt;
    rrt.setParams(params);
    rrt.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    check_result(rrt.plan(request));
}

TEST(BiRRTStarTest, RejectsNonFiniteOrOutOfLimitInput) {
    BiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());

    {
        JointConfig q_nan = JointConfig::Zero();
        q_nan[2] = std::nan("");
        auto req = makeRequest(q_nan, JointConfig::Zero());
        PlanResult r = planner.plan(req);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.failure_code, PlanningFailureCode::kInvalidInput);
    }
    {
        JointConfig q_oor = JointConfig::Zero();
        q_oor[0] = 999.0;
        auto req = makeRequest(JointConfig::Zero(), q_oor);
        PlanResult r = planner.plan(req);
        EXPECT_FALSE(r.success);
        EXPECT_EQ(r.failure_code, PlanningFailureCode::kInvalidInput);
    }
}

TEST(BiRRTStarTest, StartCollisionFailsBeforeSearch) {
    JointConfig q_start = JointConfig::Zero();
    q_start[0] = 0.2;
    auto req = makeRequest(q_start, JointConfig::Zero());

    BiRRTStar planner;
    planner.setCollisionChecker(std::make_shared<AlwaysInvalidCollision>());
    PlanResult r = planner.plan(req);
    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_NE(r.message.find("start"), std::string::npos);
}

TEST(BiRRTStarTest, GoalCollisionFailsBeforeSearch) {
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[1] = 0.3;
    auto req = makeRequest(q_start, q_goal);

    auto checker = std::make_shared<GoalOnlyInvalidCollision>(q_goal);
    BiRRTStar planner;
    planner.setCollisionChecker(checker);
    PlanResult r = planner.plan(req);

    EXPECT_FALSE(r.success);
    EXPECT_EQ(r.failure_code, PlanningFailureCode::kGoalNotReached);
    EXPECT_NE(r.message.find("goal"), std::string::npos);
    // Must not have entered the search loop — no motion checks issued.
    EXPECT_EQ(checker->motionCalls(), 0);
}

TEST(BiRRTStarTest, RequestSeedReproducesPath) {
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.3;
    auto req = makeRequest(q_start, q_goal);

    BiRRTStar planner1, planner2;
    planner1.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    planner2.setCollisionChecker(std::make_shared<AlwaysValidCollision>());

    req.random_seed = 42;
    PlanResult r1 = planner1.plan(req);
    PlanResult r2 = planner2.plan(req);

    ASSERT_EQ(r1.success, r2.success);
    if (r1.success) {
        EXPECT_EQ(r1.path.size(), r2.path.size());
        EXPECT_NEAR(r1.path_cost, r2.path_cost, 1e-10);
    }
}

TEST(BiRRTStarTest, BridgeConnectionPathIsFullyValidated) {
    // Force a multi-step connection and verify returned bridge segments.
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.8;

    auto req = makeRequest(q_start, q_goal);
    req.random_seed = 11;

    PlanningParams params;
    params.max_iterations = 20;
    params.max_step = 0.2;
    params.connect_goal_bias = 1.0;
    params.goal_bias = 1.0;

    BiRRTStar planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<ShortOnlyCollision>());
    PlanResult result = planner.plan(req);

    ASSERT_TRUE(result.success);
    EXPECT_GT(result.path.size(), 2U);
    AlwaysValidCollision checker;
    for (size_t i = 1; i < result.path.size(); ++i) {
        double seg_dist = (result.path[i] - result.path[i - 1]).norm();
        EXPECT_GT(seg_dist, 0.0) << "zero-length segment at " << (i - 1);
        EXPECT_TRUE(checker.isMotionValid(result.path[i - 1], result.path[i],
                                          params.validation_distance));
    }
}

TEST(BiRRTStarTest, RejectedBridgeDoesNotLeavePartialAcceptedPath) {
    // Use a checker where all bridge segments pass but the final edge
    // to the other tree's goal node is rejected. The planner must not
    // return a success with an invalid final edge.
    JointConfig q_start = JointConfig::Zero();
    JointConfig q_goal = JointConfig::Zero();
    q_goal[0] = 0.5;
    auto req = makeRequest(q_start, q_goal);

    auto checker = std::make_shared<BridgeValidFinalEdgeInvalidCollision>(q_goal);
    BiRRTStar planner;
    planner.setCollisionChecker(checker);
    PlanResult result = planner.plan(req);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kGoalNotReached);
}

}  // namespace
}  // namespace fairino_planning
