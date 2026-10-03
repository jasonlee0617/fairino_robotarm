from graspnet_bringup.task.task_types import GraspState


class GraspnetStateMachine:
    """Run the GraspNet execution state machine against its ROS node."""

    def __init__(self, node):
        self.node = node

    def tick(self):
        node = self.node
        if node.active_mode != "graspnet":
            return
        try:
            if node.abort.is_set():
                return
            node._publish_state(node.current_state)

            state = node.current_state
            if state == GraspState.WAIT_READY:
                self._wait_ready()
            elif state == GraspState.PREGRASP_POSE:
                self._pregrasp_pose()
            elif state == GraspState.WAIT_G:
                self._wait_g()
            elif state == GraspState.COMPUTE:
                self._compute()
            elif state == GraspState.SELECT:
                self._select()
            elif state == GraspState.PLAN:
                self._plan()
            elif state == GraspState.PREOPEN:
                self._preopen()
            elif state == GraspState.MOVE_TO_APPROACH:
                self._move_to_approach()
            elif state == GraspState.APPROACH_TO_GRASP:
                self._approach_to_grasp()
            elif state == GraspState.CLOSE:
                self._close()
            elif state == GraspState.LIFT:
                self._lift()
            elif state == GraspState.RETURN_PREGRASP:
                self._return_pregrasp()
        except Exception as exc:
            node.get_logger().error(f"GraspNet state {node.current_state} exception: {exc}")
            if node.current_state in (GraspState.COMPUTE, GraspState.SELECT, GraspState.PLAN):
                self._inference_failed("exception")
            else:
                self._motion_failed("exception")

    def _inference_failed(self, reason: str):
        node = self.node
        message = (
            f"GraspNet inference failed ({reason}); "
            "keeping the robot at its current pregrasp state."
        )
        if reason.startswith("CANCELED:"):
            node.get_logger().info(message)
        else:
            node.get_logger().error(message)
        node._reset_task_cache()
        node._set_state(GraspState.WAIT_G)

    def _motion_failed(self, reason: str):
        node = self.node
        node.get_logger().error(
            f"GraspNet motion failed ({reason}); stopped. Press h for one pregrasp reset."
        )
        if not node.abort.is_set():
            node.abort.request_abort(f"GraspNet motion failed: {reason}", command="stop")
        node.abort.cancel_all_motion_now()
        node._fail(GraspState.FAILED)

    def _wait_ready(self):
        node = self.node
        if not node._tf_ready():
            return
        if not node.compute_client.wait_for_service(timeout_sec=0.1):
            node._publish_state("waiting_graspnet")
            return
        if not node.startup_motion_ready(timeout_sec=0.1):
            node._publish_state("waiting_moveit")
            return
        node._set_state(GraspState.PREGRASP_POSE)

    def _pregrasp_pose(self):
        node = self.node
        if node._move_to_pregrasp_pose() and node._close_gripper_at_pregrasp():
            node._g_requested = False
            node._set_state(GraspState.WAIT_G)
        else:
            self._motion_failed("pregrasp_pose_failed")

    def _wait_g(self):
        node = self.node
        if node._g_requested:
            node._g_requested = False
            node._set_state(GraspState.COMPUTE)

    def _compute(self):
        node = self.node
        node._start_seq = node._result_seq
        node._publish_state("CONFIRM")
        if not node._call_compute_service():
            # A user stop owns the recovery path.  Do not turn it into a
            # harmless inference failure while its recovery is still pending.
            if not node.abort.is_set():
                self._inference_failed(
                    getattr(node, "_last_compute_error", "") or "compute_failed"
                )
            return
        node._set_state(GraspState.SELECT)

    def _select(self):
        node = self.node
        msg, scores, metadata = node._wait_for_result(node._start_seq)
        if msg is None:
            if not node.abort.is_set():
                self._inference_failed("no_grasp_result")
            return
        node._grasp_msg = msg
        node._grasp_scores = scores
        node._grasp_metadata = metadata
        node._candidates = node._build_candidates(msg, scores, metadata)
        node._active_candidate = None
        node._set_state(GraspState.PLAN)

    def _plan(self):
        node = self.node
        candidate = node._select_executable_candidate()
        if candidate is None:
            if not node.abort.is_set():
                self._inference_failed("no_executable_grasp")
            return
        node._active_candidate = candidate
        node._publish_target(candidate.grasp)
        node._publish_selected_grasp_6d(candidate)
        node._publish_grasp_plan_6d(candidate)
        node._set_state(GraspState.PREOPEN)

    def _preopen(self):
        node = self.node
        candidate = node._require_candidate()
        if candidate is None:
            return
        positions = candidate.preopen_positions if node.use_graspnet_width else node.gripper_open_positions
        if positions is None:
            node._reject_candidate(candidate, "missing_graspnet_width")
            node._set_state(GraspState.PLAN)
            return
        if node.motion.control_gripper(
            open_gripper=False,
            positions=positions,
            action_name=(
                "Set GraspNet pre-open: commanded="
                f"({positions[0]:.4f},{positions[1]:.4f})"
            ),
            timeout_sec=90.0,
        ):
            node._set_state(GraspState.MOVE_TO_APPROACH)
        else:
            self._motion_failed("graspnet_preopen_failed")

    def _move_to_approach(self):
        node = self.node
        candidate = node._require_candidate()
        if candidate is None:
            return
        if node.motion.move_to_pose(
            candidate.approach,
            planning_client=node.ik_plugin,
            cartesian=False,
            action_name="Move to GraspNet approach",
            max_velocity=0.15,
            max_acceleration=0.15,
            joint_constraint=node.j2_constraint,
            timeout_sec=180.0,
            **node._motion_limits_kwargs(),
        ):
            node._set_state(GraspState.APPROACH_TO_GRASP)
        else:
            node._reject_candidate(candidate, "move_to_approach_execute_failed")
            self._motion_failed("move_to_approach_execute_failed")

    def _approach_to_grasp(self):
        node = self.node
        candidate = node._require_candidate()
        if candidate is None:
            return
        if node.motion.move_to_pose(
            candidate.grasp,
            planning_client=node.ik_plugin,
            cartesian=True,
            action_name="Approach to GraspNet grasp",
            max_velocity=0.02,
            max_acceleration=0.02,
            joint_constraint=node.j2_constraint,
            timeout_sec=90.0,
            **node._motion_limits_kwargs(),
        ):
            node._set_state(GraspState.CLOSE)
        else:
            node._reject_candidate(candidate, "approach_to_grasp_execute_failed")
            self._motion_failed("approach_to_grasp_execute_failed")

    def _close(self):
        node = self.node
        candidate = node._require_candidate()
        if candidate is None:
            return
        close_positions = node.gripper_close_positions
        if node.motion.control_gripper(
            open_gripper=False,
            positions=close_positions,
            action_name=(
                "Close gripper: commanded="
                f"({close_positions[0]:.4f},{close_positions[1]:.4f})"
            ),
            timeout_sec=90.0,
        ):
            node.get_logger().info(
                "✓ Close gripper done: commanded="
                f"({close_positions[0]:.4f},{close_positions[1]:.4f})"
            )
            node._set_state(GraspState.LIFT)
        else:
            self._motion_failed("close_gripper_failed")

    def _lift(self):
        node = self.node
        candidate = node._require_candidate()
        if candidate is None:
            return
        if node.motion.move_to_pose(
            candidate.lift,
            planning_client=node.ik_plugin,
            cartesian=True,
            action_name="Lift GraspNet target",
            max_velocity=0.10,
            max_acceleration=0.10,
            joint_constraint=node.j2_constraint,
            timeout_sec=90.0,
            **node._motion_limits_kwargs(),
        ):
            node._set_state(GraspState.RETURN_PREGRASP)
        else:
            self._motion_failed("lift_failed")

    def _return_pregrasp(self):
        node = self.node
        if node._move_to_pregrasp_pose() and node._close_gripper_at_pregrasp():
            node._reset_task_cache()
            node._set_state(GraspState.WAIT_G)
        else:
            self._motion_failed("return_pregrasp_failed")
