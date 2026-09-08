#include "myrobot_planning_ros/moveit_collision_checker.h"

#include <gtest/gtest.h>
#include <geometric_shapes/shapes.h>
#include <moveit/utils/robot_model_test_utils.h>

namespace fairino_planning {
namespace {

planning_scene::PlanningScenePtr makeScene() {
    moveit::core::RobotModelBuilder builder("clearance_robot", "base");
    std::vector<geometry_msgs::msg::Pose> origins(6);
    for (auto& origin : origins) {
        origin.orientation.w = 1.0;
        origin.position.x = 0.2;
    }
    builder.addChain(
        "base->l1->l2->l3->l4->l5->l6", "revolute", origins,
        urdf::Vector3(0.0, 0.0, 1.0));
    geometry_msgs::msg::Pose collision_origin;
    collision_origin.orientation.w = 1.0;
    builder.addCollisionBox("l6", {0.05, 0.05, 0.05}, collision_origin);
    builder.addGroupChain("base", "l6", "arm");

    auto scene = std::make_shared<planning_scene::PlanningScene>(builder.build());
    const auto obstacle = std::make_shared<const shapes::Box>(0.1, 0.1, 0.1);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(1.2, 0.15, 0.0);
    scene->getWorldNonConst()->addToObject("obstacle", obstacle, pose);
    return scene;
}

TEST(MoveItCollisionCheckerTest, ClearanceGradientIncreasesWorldDistance) {
    const auto scene = makeScene();
    MoveItCollisionChecker checker(scene, "arm");
    const JointConfig q = JointConfig::Zero();

    const ClearanceQueryResult initial = checker.nearestWorldClearance(q, 0.5);
    ASSERT_TRUE(initial.supported);
    ASSERT_TRUE(initial.has_obstacle);
    ASSERT_TRUE(std::isfinite(initial.signed_distance));
    ASSERT_GT(initial.joint_gradient.norm(), 1e-9);
    EXPECT_FALSE(initial.robot_link.empty());

    const JointConfig moved = q + 1e-3 * initial.joint_gradient.normalized();
    const ClearanceQueryResult next = checker.nearestWorldClearance(moved, 0.5);
    ASSERT_TRUE(next.has_obstacle);
    EXPECT_GT(next.signed_distance, initial.signed_distance);
}

}  // namespace
}  // namespace fairino_planning
