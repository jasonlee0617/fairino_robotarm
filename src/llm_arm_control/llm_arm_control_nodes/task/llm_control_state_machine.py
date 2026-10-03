"""Task-state flow for the LLM robot manipulation server."""

from enum import Enum


class LlmControlTaskState(str, Enum):
    PREGRASP_POSE = "PREGRASP_POSE"
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    PREVIEW_READY = "PREVIEW_READY"
    MOVING_TO_TARGET_ABOVE = "MOVING_TO_TARGET_ABOVE"
    MOVING_TO_TARGET = "MOVING_TO_TARGET"
    GRASPING = "GRASPING"
    LIFTING_TARGET = "LIFTING_TARGET"
    RETURNING_PREGRASP_POSE = "RETURNING_PREGRASP_POSE"
    HOLDING = "HOLDING"
    SEARCHING_BOX = "SEARCHING_BOX"
    MOVING_TO_BOX_ABOVE = "MOVING_TO_BOX_ABOVE"
    DESCEND_TO_BOX = "DESCEND_TO_BOX"
    RELEASING = "RELEASING"


class LlmControlTaskStateMachine:
    """Keep task phases here; the ROS node supplies perception and motion resources."""

    def __init__(self, node):
        self.node = node
        self._pregrasp_in_flight = False

    def _set_state(self, state):
        with self.node._lock:
            self.node._state = state.value

    def _interrupted(self, execution_epoch, goal_handle):
        if execution_epoch is not None and self.node._execution_interrupted(
            execution_epoch, goal_handle
        ):
            self.node._mark_stop_state()
            return True
        return False

    def _run(
        self,
        state,
        phase,
        run,
        goal_handle,
        index,
        count,
        execution_epoch,
        pose=None,
        message="executing",
    ):
        if self._interrupted(execution_epoch, goal_handle):
            return False
        self._set_state(state)
        if goal_handle is not None:
            self.node._feedback(goal_handle, index, count, phase, message, pose)
        if not run():
            return False
        return not self._interrupted(execution_epoch, goal_handle)

    def tick(self):
        node = self.node
        with node._lock:
            if (
                node._state != LlmControlTaskState.PREGRASP_POSE.value
                or node.abort.is_set()
                or self._pregrasp_in_flight
            ):
                return
        if not node.motion.wait_client_ready("fairino", timeout_sec=0.1):
            return
        if not node.arm_controller_ready():
            return

        with node._lock:
            if (
                node._state != LlmControlTaskState.PREGRASP_POSE.value
                or node.abort.is_set()
                or self._pregrasp_in_flight
            ):
                return
            self._pregrasp_in_flight = True

        succeeded = False
        try:
            succeeded = node._move_to_pregrasp_pose() and node._apply_gripper(0.0)
        finally:
            with node._lock:
                self._pregrasp_in_flight = False
                if succeeded and node._state == LlmControlTaskState.PREGRASP_POSE.value:
                    node._state = LlmControlTaskState.IDLE.value
        if not succeeded:
            node.get_logger().error(
                "Pregrasp motion failed; stopped. Press h for one pregrasp reset."
            )
            node._stop_for_motion_failure("pregrasp_pose_failed")

    def _pick(self, source, goal_handle, start_index, step_count, execution_epoch):
        node = self.node
        poses = node._pick_preview_poses(source)
        open_width = abs(node.open_finger_position) * 2.0
        steps = (
            (
                LlmControlTaskState.OPEN_GRIPPER,
                "OPEN_GRIPPER",
                lambda: node._apply_gripper(open_width),
                None,
            ),
            (
                LlmControlTaskState.MOVING_TO_TARGET_ABOVE,
                "MOVING_TO_TARGET_ABOVE",
                lambda: node._move_pose(
                    poses["approach_pick"], "approach_pick", False, 0.2
                ),
                poses["approach_pick"],
            ),
            (
                LlmControlTaskState.MOVING_TO_TARGET,
                "MOVING_TO_TARGET",
                lambda: node._move_pose(poses["grasp"], "grasp", True, 0.02),
                poses["grasp"],
            ),
            (
                LlmControlTaskState.GRASPING,
                "GRASPING",
                lambda: node._apply_gripper(0.0),
                None,
            ),
            (
                LlmControlTaskState.LIFTING_TARGET,
                "LIFTING_TARGET",
                lambda: node._move_pose(poses["carry"], "carry", True, 0.2),
                poses["carry"],
            ),
            (
                LlmControlTaskState.RETURNING_PREGRASP_POSE,
                "RETURNING_PREGRASP_POSE",
                lambda: node._move_to_pregrasp_pose()
                and node._apply_gripper(0.0),
                node.pregrasp_pose,
            ),
        )
        for offset, (state, phase, run, pose) in enumerate(steps, 1):
            if not self._run(
                state,
                phase,
                run,
                goal_handle,
                start_index + offset,
                step_count,
                execution_epoch,
                pose,
            ):
                with node._lock:
                    if node._held_source is not None:
                        node._state = LlmControlTaskState.HOLDING.value
                return False, f"{phase} failed"
            if state == LlmControlTaskState.GRASPING:
                with node._lock:
                    node._held_source = source
        self._set_state(LlmControlTaskState.HOLDING)
        return True, "pick complete; holding object"

    def _place(
        self,
        source,
        destination,
        goal_handle,
        start_index,
        step_count,
        execution_epoch,
    ):
        node = self.node
        self._set_state(LlmControlTaskState.SEARCHING_BOX)
        if self._interrupted(execution_epoch, goal_handle):
            return False, "task stopped before placement"
        poses = node._place_preview_poses(source, destination)
        steps = (
            (
                LlmControlTaskState.MOVING_TO_BOX_ABOVE,
                "MOVING_TO_BOX_ABOVE",
                lambda: node._move_pose(
                    poses["approach_box"], "approach_box", False, 0.2
                ),
                poses["approach_box"],
            ),
            (
                LlmControlTaskState.DESCEND_TO_BOX,
                "DESCEND_TO_BOX",
                lambda: node._move_pose(poses["release"], "release", True, 0.2),
                poses["release"],
            ),
            (
                LlmControlTaskState.RELEASING,
                "RELEASING",
                lambda: node._apply_gripper(
                    abs(node.open_finger_position) * 2.0
                ),
                None,
            ),
            (
                LlmControlTaskState.RETURNING_PREGRASP_POSE,
                "RETURNING_PREGRASP_POSE",
                lambda: node._move_to_pregrasp_pose()
                and node._apply_gripper(0.0),
                node.pregrasp_pose,
            ),
        )
        for offset, (state, phase, run, pose) in enumerate(steps, 1):
            if not self._run(
                state,
                phase,
                run,
                goal_handle,
                start_index + offset,
                step_count,
                execution_epoch,
                pose,
            ):
                if (
                    source is not None
                    and state
                    in (
                        LlmControlTaskState.MOVING_TO_BOX_ABOVE,
                        LlmControlTaskState.DESCEND_TO_BOX,
                        LlmControlTaskState.RELEASING,
                    )
                ):
                    node._mark_holding_recovery(source, destination)
                return False, f"{phase} failed"
            if state == LlmControlTaskState.RELEASING and source is not None:
                with node._lock:
                    node._clear_holding_locked()
        return True, "placed into previewed box"

    def execute_action(
        self,
        action,
        goal_handle,
        step_index,
        step_count,
        execution_epoch,
    ):
        node = self.node
        action_type = action["type"]
        if action_type == "pick":
            ok, message = self._pick(
                action["source"],
                goal_handle,
                step_index,
                step_count,
                execution_epoch,
            )
            return ok, message, 6
        if action_type == "place":
            ok, message = self._place(
                action["source"],
                action["destination"],
                goal_handle,
                step_index,
                step_count,
                execution_epoch,
            )
            return ok, message, 4
        if action_type == "pick_place":
            ok, message = self._pick(
                action["source"],
                goal_handle,
                step_index,
                step_count,
                execution_epoch,
            )
            if not ok:
                if not self._interrupted(execution_epoch, goal_handle):
                    node._mark_holding_recovery(action["source"], action["destination"])
                return ok, message, 10
            ok, message = self._place(
                action["source"],
                action["destination"],
                goal_handle,
                step_index + 6,
                step_count,
                execution_epoch,
            )
            return ok, message, 10
        if action_type == "move_relative":
            node._feedback(
                goal_handle,
                step_index + 1,
                step_count,
                action_type,
                "executing",
                action["target_pose"],
            )
            ok = node._move_pose(action["target_pose"], action_type)
            return ok, f"{action_type} {'done' if ok else 'failed'}", 1
        if action_type == "set_gripper":
            width = (
                abs(node.open_finger_position) * 2.0
                if action["state"] == "open"
                else 0.0
            )
            node._feedback(
                goal_handle,
                step_index + 1,
                step_count,
                "set_gripper",
                f"state={action['state']}, width={width:.4f} m",
            )
            ok = node._apply_gripper(width)
            if ok and action["state"] == "open":
                with node._lock:
                    node._clear_holding_locked()
            return ok, f"gripper {action['state']} {'done' if ok else 'failed'}", 1
        return False, f"unsupported action: {action_type}", 0
