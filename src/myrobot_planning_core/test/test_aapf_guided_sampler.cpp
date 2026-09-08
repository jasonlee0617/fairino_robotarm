#include <gtest/gtest.h>

#include "myrobot_planning_core/aapf/aapf_guided_sampler.h"

namespace fairino_planning {
namespace {

class ClearanceCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return state_valid; }
    bool isMotionValid(const JointConfig&, const JointConfig&, double) const override {
        ++motion_checks;
        return motion_valid;
    }
    bool supportsWorldClearance() const override { return true; }
    ClearanceQueryResult nearestWorldClearance(const JointConfig&, double) const override {
        ++clearance_queries;
        ClearanceQueryResult result;
        result.supported = true;
        result.has_obstacle = true;
        result.signed_distance = distance;
        result.joint_gradient[1] = 1.0;
        result.robot_link = "wrist3_link";
        return result;
    }

    bool state_valid = true;
    bool motion_valid = true;
    double distance = 0.12;
    mutable int motion_checks = 0;
    mutable int clearance_queries = 0;
};

class RejectPrimaryCollision final : public CollisionInterface {
public:
    bool isStateValid(const JointConfig&) const override { return true; }
    bool isMotionValid(const JointConfig& from, const JointConfig& to, double) const override {
        const JointConfig delta = to - from;
        if (delta[0] <= 1e-9) return false;
        return delta[1] / delta[0] < 0.75;
    }
    bool supportsWorldClearance() const override { return true; }
    ClearanceQueryResult nearestWorldClearance(const JointConfig&, double) const override {
        ClearanceQueryResult result;
        result.supported = true;
        result.has_obstacle = true;
        result.signed_distance = 0.12;
        result.joint_gradient[1] = 1.0;
        return result;
    }
};

TEST(AapfJointFieldTest, BlendsGoalAndClearanceAndShrinksNearObstacles) {
    JointConfig q_near = JointConfig::Zero();
    JointConfig q_target = JointConfig::Zero();
    q_target[0] = 1.0;
    ClearanceQueryResult clearance;
    clearance.supported = true;
    clearance.has_obstacle = true;
    clearance.signed_distance = 0.12;
    clearance.joint_gradient[1] = 1.0;

    const auto field = evaluateAapfJointField(
        q_near, q_target, clearance, 0.24, 0.20);

    EXPECT_NEAR(field.risk, 0.5, 1e-12);
    EXPECT_NEAR(field.primary_dir[0], std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(field.primary_dir[1], std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(field.tangent_dir.dot(field.away_dir), 0.0, 1e-12);
    EXPECT_NEAR(field.step, 0.125, 1e-12);
}

TEST(AapfGuidedSamplerTest, CachesOneClearanceQueryPerTreeNode) {
    PlanningParams params;
    params.max_step = 0.20;
    params.aapf.repulsion_range_m = 0.24;
    JointLimits limits;
    ClearanceCollision collision;
    std::mt19937 rng(7);
    AapfGuidedSampler sampler(params, limits, collision, 0.03, rng);

    RRTTree current(16), opposite(16);
    current.addNode(JointConfig::Zero(), -1, 0.0);
    JointConfig q_target = JointConfig::Zero();
    q_target[0] = 0.8;
    opposite.addNode(q_target, -1, 0.0);

    const auto first = sampler.generate(
        current, opposite, q_target, 0, 0, AapfSampleSource::kGuided);
    const auto second = sampler.generate(
        current, opposite, q_target, 0, 0, AapfSampleSource::kGuided);

    EXPECT_TRUE(first.valid_edge);
    EXPECT_TRUE(second.valid_edge);
    EXPECT_EQ(first.clearance_queries, 1);
    EXPECT_EQ(second.clearance_queries, 0);
    EXPECT_EQ(collision.clearance_queries, 1);
    EXPECT_LE((first.q_new - first.q_near).norm(), params.max_step + 1e-12);
}

TEST(AapfGuidedSamplerTest, GuidedFailureDoesNotRunGlobalFallback) {
    PlanningParams params;
    params.aapf.shrink_motion_attempts = 2;
    JointLimits limits;
    ClearanceCollision collision;
    collision.motion_valid = false;
    std::mt19937 rng(11);
    AapfGuidedSampler sampler(params, limits, collision, 0.03, rng);

    RRTTree current(16), opposite(16);
    current.addNode(JointConfig::Zero(), -1, 0.0);
    JointConfig q_target = JointConfig::Zero();
    q_target[0] = 0.8;
    opposite.addNode(q_target, -1, 0.0);

    const auto sample = sampler.generate(
        current, opposite, q_target, 0, params.aapf.stall_threshold_iters,
        AapfSampleSource::kGuided);

    EXPECT_TRUE(sample.proposal_generated);
    EXPECT_FALSE(sample.valid_edge);
    EXPECT_EQ(sample.clearance_queries, 1);
    EXPECT_GT(collision.motion_checks, 0);
}

TEST(AapfGuidedSamplerTest, TangentCandidateRetainsAwayComponent) {
    PlanningParams params;
    params.max_step = 0.20;
    JointLimits limits;
    RejectPrimaryCollision collision;
    std::mt19937 rng(12);
    AapfGuidedSampler sampler(params, limits, collision, 0.03, rng);

    RRTTree current(16), opposite(16);
    current.addNode(JointConfig::Zero(), -1, 0.0);
    JointConfig target = JointConfig::Zero();
    target[0] = 0.8;
    opposite.addNode(target, -1, 0.0);

    const auto sample = sampler.generate(
        current, opposite, target, 0, 0, AapfSampleSource::kGuided);

    ASSERT_TRUE(sample.valid_edge);
    EXPECT_GT(sample.q_new[0], 0.0);
    EXPECT_GT(sample.q_new[1], 0.0);
}

TEST(AapfGuidedSamplerTest, GlobalSamplingDoesNotQueryClearance) {
    PlanningParams params;
    params.connect_goal_bias = 0.0;
    JointLimits limits;
    ClearanceCollision collision;
    std::mt19937 rng(13);
    AapfGuidedSampler sampler(params, limits, collision, 0.03, rng);

    RRTTree current(16), opposite(16);
    current.addNode(JointConfig::Zero(), -1, 0.0);
    opposite.addNode(JointConfig::Ones() * 0.1, -1, 0.0);

    const auto sample = sampler.generate(
        current, opposite, JointConfig::Zero(), 0, 0, AapfSampleSource::kGlobal);

    EXPECT_TRUE(sample.proposal_generated);
    EXPECT_EQ(sample.clearance_queries, 0);
    EXPECT_EQ(collision.clearance_queries, 0);
}

}  // namespace
}  // namespace fairino_planning
