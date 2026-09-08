#include <gtest/gtest.h>

#include "myrobot_planning_core/algorithms/prm.h"
#include "myrobot_planning_core/collision/collision_interface.h"

namespace fairino_planning {
namespace {

class AlwaysValidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override { return true; }
};

class ShortMotionCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& lhs, const JointConfig& rhs, double) const override {
        return (lhs - rhs).norm() <= 0.22;
    }
};

class MotionBlockedCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override { return false; }
};

PlanRequestCore requestTo(double goal_position) {
    PlanRequestCore request;
    request.q_start = JointConfig::Zero();
    request.q_goal = JointConfig::Zero();
    request.q_goal[0] = goal_position;
    request.random_seed = 7;
    return request;
}

PlanningParams params() {
    PlanningParams value;
    value.max_iterations = 120;
    value.prm.k_neighbors = 4;
    return value;
}

JointLimits lineLimits() {
    JointLimits limits;
    limits.lower.setZero();
    limits.upper.setZero();
    limits.upper[0] = 0.6;
    return limits;
}

}  // namespace

TEST(PRMTest, DirectConnectionUsesNearestGoalRoot) {
    PRM planner;
    planner.setParams(params());
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    auto request = requestTo(0.2);
    JointConfig closer = JointConfig::Zero();
    closer[0] = 0.1;
    request.goal_candidates = {request.q_goal, closer};
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.sample_attempts, params().max_iterations);
    EXPECT_NEAR(result.path.back()[0], 0.1, 1e-12);
}

TEST(PRMTest, FixedPostSolutionSamplingBudgetStopsAfterExactAttemptCount) {
    auto config = params();
    config.post_solution_sample_attempts = 3;
    PRM planner;
    planner.setParams(config);
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());

    const PlanResult result = planner.plan(requestTo(0.2));

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.sample_attempts, 3);
    EXPECT_EQ(result.effort_stats.post_solution_sample_attempts, 3);
    EXPECT_TRUE(result.effort_stats.post_solution_budget_complete);
}

TEST(PRMTest, RoadmapFindsAValidatedRouteWithinSafetyBudget) {
    PRM planner;
    planner.setParams(params());
    planner.setJointLimits(lineLimits());
    planner.setCollisionChecker(std::make_shared<ShortMotionCollision>());
    const PlanResult result = planner.plan(requestTo(0.6));

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.accepted_samples, 0);
    EXPECT_GT(result.iterations, 0);
    EXPECT_NE(result.diagnostics.find("prm_queries="), std::string::npos);
    EXPECT_NEAR(result.path.front()[0], 0.0, 1e-12);
    EXPECT_NEAR(result.path.back()[0], 0.6, 1e-12);
}

TEST(PRMTest, MotionRejectedRoadmapFailsAtItsSamplingBudget) {
    PRM planner;
    auto config = params();
    config.max_iterations = 9;
    planner.setParams(config);
    planner.setCollisionChecker(std::make_shared<MotionBlockedCollision>());
    const PlanResult result = planner.plan(requestTo(0.4));

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.sample_attempts, 9);
    EXPECT_EQ(result.iterations, 9);
}

}  // namespace fairino_planning
