import time
import threading
import rclpy
from std_msgs.msg import String


class AbortManager:
    """
    把以下功能集中管理：
    - /manual_abort 回调触发 abort event
    - cancel arm/gripper 的 moveit action
    - 可中断 wait（轮询 query_state）
    - recover：停下 -> disable keepout -> go_home -> open -> close -> reset cache -> clear abort flag
    """

    def __init__(self, node, arm, gripper):
        self._node = node
        self.arm = arm
        self.gripper = gripper

        self._event = threading.Event()
        self.reason = ""
        self.command = ""
        self._hooks = {}
        self._command_hook = None
        self._command_enabled = lambda: True
        self._state_lock = threading.RLock()
        self._recovery_active = False
        self._recovery_thread = None
        self._recovery_owner_ident = None
        self._recovery_message = ""
        self._recovery_released = False
        self._cancelled_goal_tokens = set()
        callback_group = getattr(self._node, "abort_cb_group", None)
        self._motion_command_sub = self._node.create_subscription(
            String, "/motion_control/command", self.on_motion_command, 10,
            callback_group=callback_group,
        )

    # --------- basic ---------
    def is_set(self) -> bool:
        return self._event.is_set()

    def is_blocked(self) -> bool:
        with self._state_lock:
            return not (
                self._recovery_active
                and self._recovery_released
                and self._recovery_owner_ident == threading.get_ident()
            ) and (self._event.is_set() or self._recovery_active)

    def clear(self):
        with self._state_lock:
            self._event.clear()
            self.reason = ""
            self.command = ""
            self._cancelled_goal_tokens.clear()

    def recovery_active(self) -> bool:
        with self._state_lock:
            return self._recovery_active

    def recovery_message(self) -> str:
        with self._state_lock:
            return self._recovery_message

    def recovery_released(self) -> bool:
        with self._state_lock:
            return self._recovery_released

    def _set_recovery_message(self, message: str):
        with self._state_lock:
            self._recovery_message = str(message)

    def is_reset_requested(self) -> bool:
        with self._state_lock:
            return self._event.is_set() and self.command == "reset"

    def is_stop_requested(self) -> bool:
        with self._state_lock:
            return self._event.is_set() and self.command in ("stop", "manual_abort")

    def set_recovery_hooks(self, **hooks):
        self._hooks.update({k: v for k, v in hooks.items() if v is not None})

    def set_command_hook(self, hook):
        self._command_hook = hook

    def set_command_enabled(self, enabled):
        self._command_enabled = enabled

    def _notify_command(self, command: str):
        if self._command_hook is None:
            return
        try:
            self._command_hook(command)
        except Exception as exc:
            self._node.get_logger().error(f"motion command hook failed: {exc}")

    def request_abort(self, reason: str, command: str = "stop") -> bool:
        command = str(command).strip().lower() or "stop"
        with self._state_lock:
            same_stop = (
                command in ("stop", "manual_abort")
                and self.command in ("stop", "manual_abort")
            )
            if self._event.is_set() and (
                same_stop or (self.command == command and self.reason == reason)
            ):
                return False

            self.reason = reason
            self.command = command
            self._event.set()
        self._notify_command("stop" if command == "manual_abort" else command)
        self._node.get_logger().warn(f"!!! Abort requested: {reason} !!!")
        return True

    # --------- ROS callback ---------
    def on_manual_abort(self, msg):
        if not msg.data:
            return
        if self.request_abort("manual abort (/manual_abort)", command="manual_abort"):
            self.cancel_all_motion_now()

    def on_motion_command(self, msg):
        if not self._command_enabled():
            return
        command = str(msg.data).strip().lower()
        if command == "g":
            return
        if command == "stop":
            if self.request_abort("motion_control stop", command="stop"):
                self.cancel_all_motion_now()
        elif command == "resume":
            with self._state_lock:
                recovery_active = self._recovery_active
                if not recovery_active:
                    self._event.clear()
                    self.reason = ""
                    self.command = ""
                    self._cancelled_goal_tokens.clear()
            if recovery_active:
                self._node.get_logger().warn(
                    "Motion resume ignored while Home recovery is active."
                )
                return
            self._notify_command("resume")
            self._node.get_logger().info("Motion control resumed.")
        elif command == "reset":
            if self._hooks:
                if not self._prepare_registered_recovery("motion_control reset"):
                    self._node.get_logger().warn(
                        "Home recovery is already active; duplicate reset ignored."
                    )
                    return
                self.cancel_all_motion_now()
                self._start_registered_recovery()
            elif self.request_abort("motion_control reset", command="reset"):
                self.cancel_all_motion_now()
        else:
            self._node.get_logger().warn(
                "Unsupported motion command. Use stop, reset, or resume."
            )

    # --------- cancel ---------
    def _cancel_moveit(self, m, name: str):
        state_value = None
        if hasattr(m, "query_state"):
            try:
                state = m.query_state()
                state_value = state.value if hasattr(state, "value") else int(state)
            except Exception as exc:
                self._node.get_logger().warn(f"[{name}] motion state lookup failed: {exc}")

        mutex = getattr(m, "_MoveIt2__execution_mutex", None)
        gh = None
        try:
            if mutex:
                mutex.acquire()
            gh = getattr(m, "_MoveIt2__execution_goal_handle", None)
        except Exception as e:
            self._node.get_logger().warn(f"[{name}] goal_handle lookup failed: {e}")
        finally:
            if mutex:
                try:
                    mutex.release()
                except Exception:
                    pass

        if gh is not None:
            token = (id(m), id(gh))
        elif state_value is None:
            token = (id(m), "legacy")
        elif state_value == 2:
            token = (id(m), "executing-without-goal")
        else:
            # IDLE requires no cancel. REQUESTING is retried once it becomes
            # EXECUTING and exposes a goal handle.
            return

        with self._state_lock:
            if token in self._cancelled_goal_tokens:
                return
            self._cancelled_goal_tokens.add(token)

        # MoveIt's event stop and the Action cancel are each sent once for a
        # given goal handle. The wait loop can still catch REQUESTING ->
        # EXECUTING and cancel the newly accepted goal.
        try:
            m.cancel_execution()
        except Exception as e:
            self._node.get_logger().warn(f"[{name}] cancel_execution failed: {e}")
        try:
            if gh is not None:
                gh.cancel_goal_async()
        except Exception as e:
            self._node.get_logger().warn(f"[{name}] goal_handle cancel failed: {e}")

    def cancel_all_motion_now(self):
        self._node.get_logger().warn("Stopping arm & gripper...")
        try:
            self._cancel_moveit(self.arm, "arm")
        except Exception:
            pass
        try:
            self._cancel_moveit(self.gripper, "gripper")
        except Exception:
            pass

    def _wait_motion_idle(self, timeout_sec: float) -> tuple[bool, str]:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        clients = ((self.arm, "arm"), (self.gripper, "gripper"))
        while True:
            if self.is_stop_requested():
                return False, "Home recovery interrupted by stop"
            all_idle = True
            for moveit, name in clients:
                if moveit is None or not hasattr(moveit, "query_state"):
                    continue
                try:
                    state = moveit.query_state()
                    state_value = state.value if hasattr(state, "value") else int(state)
                except Exception as exc:
                    return False, f"cannot read {name} motion state: {exc}"
                if state_value != 0:
                    all_idle = False
                    self._cancel_moveit(moveit, name)
            if all_idle:
                return True, "motion stopped"
            if time.monotonic() >= deadline:
                return False, f"motion did not stop within {float(timeout_sec):g} seconds"
            time.sleep(0.02)

    def _begin_recovery(self) -> bool:
        with self._state_lock:
            if self._recovery_active:
                return False
            self._recovery_active = True
            self._recovery_owner_ident = threading.get_ident()
            self._recovery_message = "waiting for active motion to stop"
            self._recovery_released = False
            return True

    def _prepare_registered_recovery(self, reason: str) -> bool:
        with self._state_lock:
            if self._recovery_active:
                return False
            self.reason = str(reason)
            self.command = "reset"
            self._event.set()
            self._recovery_active = True
            self._recovery_owner_ident = None
            self._recovery_message = "waiting for active motion to stop"
            self._recovery_released = False
        self._notify_command("reset")
        self._node.get_logger().warn(f"!!! Abort requested: {reason} !!!")
        return True

    def _end_recovery(self):
        with self._state_lock:
            self._recovery_active = False
            self._recovery_thread = None
            self._recovery_owner_ident = None

    def _start_registered_recovery(self) -> bool:
        hooks = dict(self._hooks)
        thread = threading.Thread(
            target=self._run_registered_recovery,
            args=(hooks,),
            name="motion-reset-recovery",
            daemon=True,
        )
        with self._state_lock:
            if not self._recovery_active or self._recovery_thread is not None:
                return False
            self._recovery_thread = thread
        thread.start()
        return True

    def _run_registered_recovery(self, hooks):
        with self._state_lock:
            self._recovery_owner_ident = threading.get_ident()
        try:
            self._recover_impl(motion_already_cancelled=True, **hooks)
        finally:
            self._end_recovery()

    def shutdown_recovery(self, timeout_sec=1.0):
        with self._state_lock:
            thread = self._recovery_thread
            active = self._recovery_active
        if not active:
            return
        self.request_abort("node shutdown", command="stop")
        self.cancel_all_motion_now()
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout_sec)))

    # --------- interruptible wait ---------
    def wait_idle_or_abort(self, m, action_name: str, timeout_sec: float) -> bool:
        """
        代替 wait_until_executed()
        - abort 优先：触发时立即 cancel 并返回 False
        - 仅在动作已进入 REQUESTING/EXECUTING 后，query_state()==IDLE 时返回 motion_suceeded
        """
        t0 = time.monotonic()
        saw_active = False
        start_timeout_sec = min(float(timeout_sec), 2.0)
        while rclpy.ok():
            if self.is_blocked():
                self._node.get_logger().warn(f"ABORT while waiting: {action_name}")
                self.cancel_all_motion_now()
                return False

            try:
                st = m.query_state()
                st_val = st.value if hasattr(st, "value") else int(st)
            except Exception as exc:
                self._node.get_logger().error(f"{action_name}: cannot read motion state: {exc}")
                self.cancel_all_motion_now()
                return False

            if st_val != 0:
                saw_active = True
            elif saw_active:
                return bool(getattr(m, "motion_suceeded", False))

            elapsed_sec = time.monotonic() - t0
            if not saw_active and elapsed_sec >= start_timeout_sec:
                self._node.get_logger().error(f"{action_name} did not start -> force stop")
                self.cancel_all_motion_now()
                return False

            if elapsed_sec > timeout_sec:
                self._node.get_logger().error(f"{action_name} timeout -> force stop")
                self.cancel_all_motion_now()
                try:
                    m.force_reset_executing_state()
                except Exception:
                    pass
                return False

            time.sleep(0.02)

        return False

    # --------- recovery ---------
    def _call_recovery_complete(self, recovery_complete_fn, ok: bool):
        if recovery_complete_fn is None:
            return
        try:
            recovery_complete_fn(bool(ok))
        except Exception as exc:
            self._node.get_logger().error(f"recovery completion hook failed: {exc}")

    def _interrupt_recovery(self, message: str, recovery_complete_fn=None) -> bool:
        self._set_recovery_message(message)
        self._node.get_logger().warn(message)
        self._call_recovery_complete(recovery_complete_fn, False)
        return False

    def _fail_recovery(self, message: str, recovery_complete_fn=None) -> bool:
        detail = f"motion_control reset failed: {message}"
        with self._state_lock:
            self.reason = detail
            self.command = "reset"
            self._event.set()
            self._recovery_message = detail
        self._node.get_logger().error(detail)
        self._call_recovery_complete(recovery_complete_fn, False)
        return False

    def recover(
        self,
        keepout=None,
        open_gripper_fn=None,
        close_gripper_fn=None,
        go_home_fn=None,
        reset_fn=None,
        restore_arm_limits_fn=None,
        recovery_complete_fn=None,
        wait_task_stopped_fn=None,
        stop_timeout_sec=5.0,
    ) -> bool:
        """Synchronously execute one stop -> pregrasp -> open -> close recovery chain."""
        if not self._begin_recovery():
            self._node.get_logger().warn("Home recovery is already active.")
            return False
        try:
            return self._recover_impl(
                keepout=keepout,
                open_gripper_fn=open_gripper_fn,
                close_gripper_fn=close_gripper_fn,
                go_home_fn=go_home_fn,
                reset_fn=reset_fn,
                restore_arm_limits_fn=restore_arm_limits_fn,
                recovery_complete_fn=recovery_complete_fn,
                wait_task_stopped_fn=wait_task_stopped_fn,
                stop_timeout_sec=stop_timeout_sec,
                motion_already_cancelled=False,
            )
        finally:
            self._end_recovery()

    def _recover_impl(
        self,
        keepout=None,
        open_gripper_fn=None,
        close_gripper_fn=None,
        go_home_fn=None,
        reset_fn=None,
        restore_arm_limits_fn=None,
        recovery_complete_fn=None,
        wait_task_stopped_fn=None,
        stop_timeout_sec=5.0,
        motion_already_cancelled=False,
    ) -> bool:
        self._node.get_logger().warn(f"=== RECOVER FROM ABORT: {self.reason} ===")

        # 1) Stop the MoveIt actions and wait for controller-side completion.
        if not motion_already_cancelled:
            self.cancel_all_motion_now()
        self._set_recovery_message("waiting for active motion to stop")
        stopped, message = self._wait_motion_idle(stop_timeout_sec)
        if not stopped:
            if self.is_stop_requested():
                return self._interrupt_recovery(message, recovery_complete_fn)
            return self._fail_recovery(message, recovery_complete_fn)

        # 2) The owner action must unwind before recovery sends a new goal.
        if wait_task_stopped_fn is not None:
            self._set_recovery_message("waiting for the active task to exit")
            try:
                task_stopped = bool(wait_task_stopped_fn(float(stop_timeout_sec)))
            except TypeError:
                task_stopped = bool(wait_task_stopped_fn())
            except Exception as exc:
                return self._fail_recovery(
                    f"cannot confirm task exit: {exc}", recovery_complete_fn
                )
            if not task_stopped:
                if self.is_stop_requested():
                    return self._interrupt_recovery(
                        "Home recovery interrupted by stop", recovery_complete_fn
                    )
                return self._fail_recovery(
                    f"active task did not exit within {float(stop_timeout_sec):g} seconds",
                    recovery_complete_fn,
                )

        with self._state_lock:
            if self.command != "reset" or not self._event.is_set():
                return self._interrupt_recovery(
                    "Home recovery interrupted before returning to pregrasp",
                    recovery_complete_fn,
                )

        # 3) Disable optional keepout constraints.
        try:
            if keepout is not None and getattr(keepout, "enabled", False):
                keepout.disable()
        except Exception as exc:
            self._node.get_logger().warn(f"disable keepout during reset failed: {exc}")

        # 4) The event remains set, but the recovery owner may now execute the
        # explicitly sequenced physical recovery motions.
        with self._state_lock:
            self._recovery_released = True

        # 5) Restore optional arm limits before pregrasp planning.
        try:
            if restore_arm_limits_fn is not None:
                restore_arm_limits_fn()
        except Exception as exc:
            self._node.get_logger().warn(f"restore arm limits during reset failed: {exc}")

        # 6) Return to the caller's pregrasp pose before moving the gripper.
        self._set_recovery_message("returning to pregrasp")
        if go_home_fn is None:
            return self._fail_recovery("pregrasp hook is unavailable", recovery_complete_fn)
        try:
            ok_home = bool(go_home_fn())
        except Exception as exc:
            ok_home = False
            message = f"pregrasp raised: {exc}"
        else:
            message = "pregrasp failed"
        if self.is_stop_requested():
            return self._interrupt_recovery(
                "Home recovery interrupted by stop while returning to pregrasp",
                recovery_complete_fn,
            )
        if not ok_home:
            return self._fail_recovery(message, recovery_complete_fn)

        # 7) Open after the arm is safely at pregrasp.
        self._set_recovery_message("opening gripper")
        if open_gripper_fn is None:
            return self._fail_recovery("open-gripper hook is unavailable", recovery_complete_fn)
        try:
            opened = bool(open_gripper_fn())
        except Exception as exc:
            opened = False
            message = f"open gripper raised: {exc}"
        else:
            message = "open gripper failed"
        if self.is_stop_requested():
            return self._interrupt_recovery(
                "Home recovery interrupted by stop while opening gripper",
                recovery_complete_fn,
            )
        if not opened:
            return self._fail_recovery(message, recovery_complete_fn)

        # 8) Close at pregrasp. Existing callers without a close hook retain
        # their legacy behavior; every grasping entry point supplies one.
        if close_gripper_fn is not None:
            self._set_recovery_message("closing gripper")
            try:
                closed = bool(close_gripper_fn())
            except Exception as exc:
                closed = False
                message = f"close gripper raised: {exc}"
            else:
                message = "close gripper failed"
            if self.is_stop_requested():
                return self._interrupt_recovery(
                    "Home recovery interrupted by stop while closing gripper",
                    recovery_complete_fn,
                )
            if not closed:
                return self._fail_recovery(message, recovery_complete_fn)

        # 9) Reset owner caches after the physical recovery succeeds.
        try:
            if reset_fn is not None:
                reset_fn()
        except Exception as exc:
            return self._fail_recovery(f"reset cache failed: {exc}", recovery_complete_fn)

        self.clear()
        self._set_recovery_message("pregrasp reset completed")
        self._call_recovery_complete(recovery_complete_fn, True)
        return True
