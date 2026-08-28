#!/usr/bin/env python3
"""Regression tests for MoveItMotion.wait_client_ready()."""

import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from types import SimpleNamespace


class _FakeService:
    def __init__(self, available):
        self._available = available

    def wait_for_service(self, timeout_sec):
        return self._available


class _FakeArm:
    def __init__(self, has_service, service_available=True):
        if has_service:
            self._plan_kinematic_path_service = _FakeService(service_available)
        else:
            pass  # no _plan_kinematic_path_service attribute

    def wait_for_server(self, timeout_sec):
        raise NotImplementedError("must not be called")


class _BlockedAbort:
    def is_blocked(self):
        return True


class _Future:
    def __init__(self, done=True):
        self._done = done

    def done(self):
        return self._done


class _ConfigurationMoveIt(_FakeArm):
    def __init__(self, planning_done=True, trajectory=object()):
        super().__init__(has_service=False)
        self.future = _Future(planning_done)
        self.trajectory = trajectory
        self.planned_positions = None
        self.executed = None

    def plan_async(self, joint_positions):
        self.planned_positions = joint_positions
        return self.future

    def get_trajectory(self, future):
        assert future is self.future
        return self.trajectory

    def execute(self, trajectory):
        self.executed = trajectory


class _GripperMoveIt(_ConfigurationMoveIt):
    def __init__(self, positions, trajectory):
        super().__init__(trajectory=trajectory)
        self.joint_state = SimpleNamespace(
            name=["finger1_joint", "finger2_joint"], position=list(positions)
        )


class _PoseMoveIt(_FakeArm):
    def __init__(self, trajectory):
        super().__init__(has_service=False)
        self.pipeline_id = "fairino"
        self.planner_id = "tube_birrt*"
        self.trajectory = trajectory
        self.executed = None
        self.plan_kwargs = None

    def clear_path_constraints(self):
        pass

    def plan(self, *_args, **kwargs):
        self.plan_kwargs = kwargs
        return self.trajectory

    def _retime_trajectory_if_needed(self, trajectory, *, cartesian):
        return trajectory

    def execute(self, trajectory):
        self.executed = trajectory


def _pose():
    return PoseStamped()


def _trajectory(stamp_sec):
    trajectory = JointTrajectory()
    trajectory.header.stamp.sec = stamp_sec
    trajectory.header.stamp.nanosec = 123
    return trajectory


@pytest.fixture(scope="module")
def ros_node():
    rclpy.init(args=[])
    node = Node("_test_motion")
    yield node
    node.destroy_node()
    rclpy.shutdown()


class TestWaitClientReady:
    def test_service_ready_returns_true(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=True, service_available=True)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        assert m.wait_client_ready() is True

    def test_service_timeout_returns_false(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=True, service_available=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        assert m.wait_client_ready(timeout_sec=0.1) is False

    def test_no_service_attr_returns_true_no_exception(self, ros_node):
        """Arm without _plan_kinematic_path_service should not crash."""
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
        )
        # must not raise AttributeError
        result = m.wait_client_ready()
        assert result is True

    def test_blocked_motion_rejects_joint_move(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _FakeArm(has_service=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
            abort=_BlockedAbort(),
        )
        assert m.move_to_joints([0.0], timeout_sec=0.1) is False

    def test_joint_move_uses_bounded_async_plan_then_execute(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _ConfigurationMoveIt()
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
            action_delay=0.0,
        )

        assert m.move_to_joints([0.1], timeout_sec=0.1)
        assert arm.planned_positions == [0.1]
        assert arm.executed is arm.trajectory

    def test_joint_planning_respects_total_timeout(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        arm = _ConfigurationMoveIt(planning_done=False)
        m = MoveItMotion(
            node=ros_node,
            arm_clients={"fairino": arm},
            gripper=None,
            pose_tools=None,
            action_delay=0.0,
        )

        assert not m.move_to_joints([0.1], timeout_sec=0.0)
        assert arm.executed is None

    def test_closed_gripper_within_one_millimeter_skips_duplicate_motion(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        gripper = _GripperMoveIt((0.0, 0.0), JointTrajectory())
        motion = MoveItMotion(node=ros_node, arm_clients={}, gripper=gripper, action_delay=0.0)

        assert motion.control_gripper(False, positions=(0.001, -0.001))
        assert gripper.planned_positions is None
        assert gripper.executed is None

    def test_non_increasing_gripper_trajectory_is_rejected_when_not_at_target(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        trajectory = JointTrajectory()
        trajectory.points = [JointTrajectoryPoint(), JointTrajectoryPoint()]
        gripper = _GripperMoveIt((0.01, -0.01), trajectory)
        motion = MoveItMotion(node=ros_node, arm_clients={}, gripper=gripper, action_delay=0.0)

        assert not motion.control_gripper(False, positions=(0.001, -0.001))
        assert gripper.executed is None

    def test_cartesian_execution_clears_stale_trajectory_stamp(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        trajectory = _trajectory(37)
        arm = _PoseMoveIt(trajectory)
        motion = MoveItMotion(node=ros_node, arm_clients={"fairino": arm}, action_delay=0.0)
        motion._plan_fairino_cartesian = lambda **_kwargs: trajectory
        motion._wait = lambda *_args: True

        assert motion.move_to_pose(_pose(), cartesian=True, timeout_sec=0.1)
        assert arm.executed is trajectory
        assert arm.executed.header.stamp.sec == 0
        assert arm.executed.header.stamp.nanosec == 0

    def test_global_execution_preserves_trajectory_stamp(self, ros_node):
        from myrobot_common.planning.motion_executor import MoveItMotion
        trajectory = _trajectory(37)
        arm = _PoseMoveIt(trajectory)
        motion = MoveItMotion(node=ros_node, arm_clients={"fairino": arm}, action_delay=0.0)
        motion._wait = lambda *_args: True

        assert motion.move_to_pose(_pose(), cartesian=False, timeout_sec=0.1)
        assert arm.executed is trajectory
        assert arm.executed.header.stamp.sec == 37
        assert arm.executed.header.stamp.nanosec == 123

    def test_plan_to_pose_forwards_tolerances_and_start_state(self, ros_node):
        from sensor_msgs.msg import JointState
        from myrobot_common.planning.motion_executor import MoveItMotion

        trajectory = JointTrajectory()
        arm = _PoseMoveIt(trajectory)
        motion = MoveItMotion(node=ros_node, arm_clients={"fairino": arm}, action_delay=0.0)
        start = JointState(name=["j1"], position=[0.2])

        assert motion.plan_to_pose(
            _pose(),
            position_tolerance=0.004,
            orientation_tolerance=0.03,
            start_joint_state=start,
        ) is trajectory
        assert arm.plan_kwargs["tolerance_position"] == 0.004
        assert arm.plan_kwargs["tolerance_orientation"] == 0.03
        assert arm.plan_kwargs["start_joint_state"] is start


def test_ik_client_and_pipeline_are_independent_valid_choices():
    from myrobot_common.planning.motion_executor import PlannerSwitch

    for ik, pipeline, planner in (
        ("fairino", "fairino", "tube_birrt*"),
        ("fairino", "ompl", "RRTConnectFast"),
        ("kdl", "fairino", "tube_birrt*"),
        ("kdl", "ompl", "RRTConnectFast"),
    ):
        assert PlannerSwitch.normalize_ik(ik) == ik
        assert PlannerSwitch.is_valid(pipeline, planner)
