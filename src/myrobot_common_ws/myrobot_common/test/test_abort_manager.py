#!/usr/bin/env python3
import threading
import time
from unittest.mock import patch

from std_msgs.msg import Bool
from std_msgs.msg import String

from myrobot_common.task.abort_manager import AbortManager


class _Logger:
    def warn(self, msg):
        pass

    def info(self, msg):
        pass

    def error(self, msg):
        pass


class _Node:
    def create_subscription(self, *args, **kwargs):
        return object()

    def get_logger(self):
        return _Logger()


class _MoveIt:
    motion_suceeded = True

    def __init__(self):
        self.cancelled = 0

    def cancel_execution(self):
        self.cancelled += 1


class _StateMoveIt(_MoveIt):
    def __init__(self, states, succeeded=True):
        super().__init__()
        self.states = iter(states)
        self.motion_suceeded = succeeded

    def query_state(self):
        return next(self.states)


def _wait_recovery(abort, timeout_sec=1.0):
    deadline = time.monotonic() + timeout_sec
    while abort.recovery_active() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not abort.recovery_active()


def test_motion_commands_update_state_and_cancel():
    arm = _MoveIt()
    gripper = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=gripper)

    abort.on_motion_command(String(data="stop"))
    assert abort.is_set()
    assert abort.is_blocked()
    assert abort.is_stop_requested()
    assert not abort.is_reset_requested()
    assert arm.cancelled == 1
    assert gripper.cancelled == 1

    abort.on_motion_command(String(data="resume"))
    assert not abort.is_blocked()
    assert abort.command == ""


def test_duplicate_stop_and_manual_abort_cancel_once():
    arm = _MoveIt()
    gripper = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=gripper)

    abort.on_motion_command(String(data="stop"))
    abort.on_manual_abort(Bool(data=True))
    abort.on_motion_command(String(data="stop"))

    assert arm.cancelled == 1
    assert gripper.cancelled == 1


def test_pause_and_clear_are_unsupported():
    arm = _MoveIt()
    gripper = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=gripper)

    abort.on_motion_command(String(data="pause"))
    abort.on_motion_command(String(data="clear"))

    assert not abort.is_blocked()
    assert arm.cancelled == 0
    assert gripper.cancelled == 0


def test_reset_runs_registered_hooks_and_clears_state():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        open_gripper_fn=lambda: calls.append("open") or True,
        close_gripper_fn=lambda: calls.append("close") or True,
        go_home_fn=lambda: calls.append("home") or True,
        reset_fn=lambda: calls.append("reset"),
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == ["home", "open", "close", "reset"]
    assert not abort.is_blocked()


def test_command_hook_invalidates_before_reset_recovery():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_command_hook(lambda command: calls.append(f"command:{command}"))
    abort.set_recovery_hooks(
        open_gripper_fn=lambda: calls.append("open") or True,
        close_gripper_fn=lambda: calls.append("close") or True,
        go_home_fn=lambda: calls.append("home") or True,
        recovery_complete_fn=lambda ok: calls.append(f"complete:{ok}"),
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == ["command:reset", "home", "open", "close", "complete:True"]


def test_command_hook_observes_stop_and_resume():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_command_hook(calls.append)

    abort.on_motion_command(String(data="stop"))
    abort.on_motion_command(String(data="resume"))

    assert calls == ["stop", "resume"]


def test_failed_reset_reports_completion_while_remaining_blocked():
    outcomes = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        go_home_fn=lambda: False,
        recovery_complete_fn=outcomes.append,
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert outcomes == [False]
    assert abort.is_blocked()


def test_reset_command_is_distinct_from_stop():
    arm = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=_MoveIt())

    abort.on_motion_command(String(data="reset"))

    assert abort.is_set()
    assert abort.is_reset_requested()
    assert not abort.is_stop_requested()
    assert arm.cancelled == 1


def test_failed_reset_keeps_motion_blocked():
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(go_home_fn=lambda: False)

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert abort.is_blocked()
    assert abort.reason.startswith("motion_control reset failed:")


def test_reset_waits_for_task_exit_then_runs_pregrasp_open_close():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        wait_task_stopped_fn=lambda timeout: calls.append(("task_idle", timeout)) or True,
        open_gripper_fn=lambda: calls.append("open") or True,
        close_gripper_fn=lambda: calls.append("close") or True,
        go_home_fn=lambda: calls.append("home") or True,
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == [("task_idle", 5.0), "home", "open", "close"]
    assert abort.recovery_message() == "pregrasp reset completed"


def test_duplicate_reset_runs_one_recovery_chain():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def open_gripper():
        calls.append("open")
        started.set()
        release.wait(0.5)
        return True

    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(open_gripper_fn=open_gripper, go_home_fn=lambda: True)

    abort.on_motion_command(String(data="reset"))
    assert started.wait(0.5)
    abort.on_motion_command(String(data="reset"))
    release.set()
    _wait_recovery(abort)

    assert calls == ["open"]


def test_resume_is_ignored_during_recovery():
    started = threading.Event()
    release = threading.Event()
    commands = []

    def open_gripper():
        started.set()
        release.wait(0.5)
        return True

    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_command_hook(commands.append)
    abort.set_recovery_hooks(open_gripper_fn=open_gripper, go_home_fn=lambda: True)

    abort.on_motion_command(String(data="reset"))
    assert started.wait(0.5)
    abort.on_motion_command(String(data="resume"))
    release.set()
    _wait_recovery(abort)

    assert commands == ["reset"]


def test_stop_interrupts_recovery_before_close():
    started = threading.Event()
    release = threading.Event()
    calls = []
    outcomes = []

    def open_gripper():
        calls.append("open")
        started.set()
        release.wait(0.5)
        return True

    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        open_gripper_fn=open_gripper,
        go_home_fn=lambda: calls.append("home") or True,
        recovery_complete_fn=outcomes.append,
    )

    abort.on_motion_command(String(data="reset"))
    assert started.wait(0.5)
    abort.on_motion_command(String(data="stop"))
    release.set()
    _wait_recovery(abort)

    assert calls == ["home", "open"]
    assert outcomes == [False]
    assert abort.is_stop_requested()
    assert "interrupted by stop" in abort.recovery_message()


def test_open_failure_keeps_reset_blocked_after_pregrasp():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        open_gripper_fn=lambda: calls.append("open") or False,
        go_home_fn=lambda: calls.append("home") or True,
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == ["home", "open"]
    assert abort.is_reset_requested()
    assert "open gripper failed" in abort.recovery_message()


def test_stop_confirmation_timeout_keeps_reset_blocked():
    class AlwaysExecuting(_MoveIt):
        def query_state(self):
            return 2

    calls = []
    abort = AbortManager(_Node(), arm=AlwaysExecuting(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        open_gripper_fn=lambda: calls.append("open") or True,
        go_home_fn=lambda: True,
        stop_timeout_sec=0.02,
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == []
    assert abort.is_reset_requested()
    assert "did not stop within" in abort.recovery_message()


def test_resume_cannot_clear_reset_between_latch_and_worker_start():
    cancel_started = threading.Event()
    release_cancel = threading.Event()
    commands = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_command_hook(commands.append)
    abort.set_recovery_hooks(open_gripper_fn=lambda: True, go_home_fn=lambda: True)

    def blocking_cancel():
        cancel_started.set()
        release_cancel.wait(0.5)

    abort.cancel_all_motion_now = blocking_cancel
    reset_thread = threading.Thread(
        target=abort.on_motion_command, args=(String(data="reset"),)
    )
    reset_thread.start()
    assert cancel_started.wait(0.5)

    abort.on_motion_command(String(data="resume"))
    assert commands == ["reset"]
    assert abort.recovery_active()
    assert abort.is_blocked()

    release_cancel.set()
    reset_thread.join(0.5)
    _wait_recovery(abort)


def test_recovery_owner_is_allowed_but_other_threads_remain_blocked():
    open_started = threading.Event()
    release_open = threading.Event()
    owner_blocked = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())

    def open_gripper():
        owner_blocked.append(abort.is_blocked())
        open_started.set()
        release_open.wait(0.5)
        return True

    abort.set_recovery_hooks(open_gripper_fn=open_gripper, go_home_fn=lambda: True)
    abort.on_motion_command(String(data="reset"))
    assert open_started.wait(0.5)

    assert abort.is_set()
    assert abort.is_blocked()
    assert owner_blocked == [False]

    release_open.set()
    _wait_recovery(abort)


def test_close_failure_never_resets_or_clears_abort():
    calls = []
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())
    abort.set_recovery_hooks(
        go_home_fn=lambda: calls.append("home") or True,
        open_gripper_fn=lambda: calls.append("open") or True,
        close_gripper_fn=lambda: calls.append("close") or False,
        reset_fn=lambda: calls.append("reset"),
    )

    abort.on_motion_command(String(data="reset"))
    _wait_recovery(abort)

    assert calls == ["home", "open", "close"]
    assert abort.is_reset_requested()
    assert abort.is_blocked()


def test_same_moveit_goal_is_cancelled_once():
    class GoalHandle:
        def __init__(self):
            self.cancelled = 0

        def cancel_goal_async(self):
            self.cancelled += 1

    class ActiveMoveIt:
        def __init__(self):
            self.cancelled = 0
            self._MoveIt2__execution_mutex = threading.Lock()
            self._MoveIt2__execution_goal_handle = GoalHandle()

        def query_state(self):
            return 2

        def cancel_execution(self):
            self.cancelled += 1

    arm = ActiveMoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=None)

    abort.cancel_all_motion_now()
    abort.cancel_all_motion_now()

    assert arm.cancelled == 1
    assert arm._MoveIt2__execution_goal_handle.cancelled == 1


def test_shutdown_interrupts_and_joins_recovery_worker():
    started = threading.Event()
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())

    def open_gripper():
        started.set()
        deadline = time.monotonic() + 0.5
        while not abort.is_stop_requested() and time.monotonic() < deadline:
            time.sleep(0.005)
        return False

    abort.set_recovery_hooks(open_gripper_fn=open_gripper, go_home_fn=lambda: True)
    abort.on_motion_command(String(data="reset"))
    assert started.wait(0.5)

    abort.shutdown_recovery(timeout_sec=0.5)

    assert not abort.recovery_active()
    assert abort.is_stop_requested()


def test_reset_without_hooks_leaves_abort_for_owner_recovery():
    arm = _MoveIt()
    abort = AbortManager(_Node(), arm=arm, gripper=_MoveIt())

    abort.on_motion_command(String(data="reset"))

    assert abort.is_set()
    assert arm.cancelled == 1


@patch("myrobot_common.task.abort_manager.rclpy.ok", return_value=True)
def test_wait_requires_an_active_motion_before_idle_success(_ok):
    abort = AbortManager(_Node(), arm=_MoveIt(), gripper=_MoveIt())

    assert not abort.wait_idle_or_abort(_StateMoveIt([0]), "idle", timeout_sec=0.0)
    assert abort.wait_idle_or_abort(_StateMoveIt([1, 2, 0]), "motion", timeout_sec=1.0)
