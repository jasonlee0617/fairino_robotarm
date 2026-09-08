#include <gtest/gtest.h>

#include "myrobot_planning_core/algorithms/informed_rrt_star.h"
#include "myrobot_planning_core/algorithms/rrt_star.h"
#include "myrobot_planning_core/collision/collision_interface.h"

namespace fairino_planning {
namespace {

class ShortMotionCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& lhs, const JointConfig& rhs, double) const override {
        return (lhs - rhs).norm() <= 0.21;
    }
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
    value.max_iterations = 8;
    value.max_step = 0.2;
    value.goal_threshold = 0.21;
    value.goal_bias = 1.0;
    return value;
}

}  // namespace

TEST(InformedRRTStarTest, FirstSolutionMatchesRrtStarBeforeInformedSampling) {
    const auto request = requestTo(0.4);
    const auto collision = std::make_shared<ShortMotionCollision>();

    RRTStar baseline;
    baseline.setParams(params());
    baseline.setCollisionChecker(collision);
    const PlanResult baseline_result = baseline.plan(request);

    InformedRRTStar informed;
    informed.setParams(params());
    informed.setCollisionChecker(collision);
    const PlanResult informed_result = informed.plan(request);

    ASSERT_TRUE(baseline_result.success) << baseline_result.message;
    ASSERT_TRUE(informed_result.success) << informed_result.message;
    EXPECT_EQ(informed_result.iterations, baseline_result.iterations);
    EXPECT_NEAR(informed_result.path_cost, baseline_result.path_cost, 1e-12);
}

TEST(InformedRRTStarTest, AnytimeSearchKeepsFinalPathValid) {
    InformedRRTStar planner;
    planner.setParams(params());
    planner.setCollisionChecker(std::make_shared<ShortMotionCollision>());
    const PlanResult result = planner.plan(requestTo(0.4));

    ASSERT_TRUE(result.success) << result.message;
    EXPECT_GT(result.iterations, 1);
    // The lower-bound root has no non-degenerate ellipsoid; the post-solution
    // budget still records both constrained sampling attempts.
    EXPECT_NEAR(result.path.back()[0], 0.4, 1e-12);
}

}  // namespace fairino_planning
