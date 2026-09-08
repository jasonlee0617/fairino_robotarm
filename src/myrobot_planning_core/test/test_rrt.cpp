#include <gtest/gtest.h>

#include "myrobot_planning_core/algorithms/rrt.h"
#include "myrobot_planning_core/collision/collision_interface.h"

namespace fairino_planning {
namespace {

class AlwaysValidCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override { return true; }
};

class MotionBlockedCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override { return false; }
};

class ShortMotionCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& a, const JointConfig& b, double) const override {
        return (a - b).norm() <= 0.11;
    }
};

PlanRequestCore requestTo(const JointConfig& goal) {
    PlanRequestCore request;
    request.q_start = JointConfig::Zero();
    request.q_goal = goal;
    request.random_seed = 7;
    return request;
}

}  // namespace

TEST(RRTTest, DirectConnectionUsesNearestValidGoalRootWithoutSampling) {
    JointConfig preferred = JointConfig::Zero();
    JointConfig alternate = JointConfig::Zero();
    preferred[0] = 0.20;
    alternate[0] = 0.05;
    auto request = requestTo(preferred);
    request.goal_candidates = {preferred, alternate};

    RRT planner;
    planner.setCollisionChecker(std::make_shared<AlwaysValidCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_NEAR((result.path.back() - alternate).norm(), 0.0, 1e-10);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.sample_attempts, 0);
    EXPECT_EQ(result.accepted_samples, 0);
}

TEST(RRTTest, FirstSolutionStopsAfterTheFirstStrictGoalConnection) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.20;
    auto request = requestTo(goal);
    PlanningParams params;
    params.max_iterations = 16;
    params.max_step = 0.10;
    params.goal_threshold = 0.11;
    params.goal_bias = 1.0;

    RRT planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<ShortMotionCollision>());
    const PlanResult result = planner.plan(request);

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.sample_attempts, 1);
    EXPECT_EQ(result.accepted_samples, 1);
}

TEST(RRTTest, CollisionRejectionDoesNotCreateTreeSamples) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto request = requestTo(goal);
    PlanningParams params;
    params.max_iterations = 3;
    params.goal_bias = 1.0;

    RRT planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<MotionBlockedCollision>());
    const PlanResult result = planner.plan(request);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.sample_attempts, 3);
    EXPECT_EQ(result.accepted_samples, 0);
}

TEST(RRTTest, CooperativeCancellationStopsSearch) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto request = requestTo(goal);
    request.cancel_requested = []() { return true; };
    PlanningParams params;
    params.max_iterations = 50000;
    RRT planner;
    planner.setParams(params);
    planner.setCollisionChecker(std::make_shared<MotionBlockedCollision>());
    const PlanResult result = planner.plan(request);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.stop_reason, "cancelled");
}

TEST(RRTTest, AbsoluteDeadlineReturnsNoSolution) {
    JointConfig goal = JointConfig::Zero();
    goal[0] = 0.40;
    auto request = requestTo(goal);
    request.deadline = std::chrono::steady_clock::now();
    RRT planner;
    planner.setCollisionChecker(std::make_shared<MotionBlockedCollision>());
    const PlanResult result = planner.plan(request);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure_code, PlanningFailureCode::kTimeout);
    EXPECT_EQ(result.stop_reason, "deadline_no_solution");
}

}  // namespace fairino_planning
