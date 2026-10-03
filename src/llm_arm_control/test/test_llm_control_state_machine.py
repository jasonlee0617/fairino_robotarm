from contextlib import nullcontext
import threading
from types import SimpleNamespace

from llm_arm_control_nodes.task.llm_control_state_machine import (
    LlmControlTaskState,
    LlmControlTaskStateMachine,
)


def test_pregrasp_state_transitions_to_idle_when_motion_is_ready():
    node = SimpleNamespace(
        _state=LlmControlTaskState.PREGRASP_POSE.value,
        abort=SimpleNamespace(is_set=lambda: False),
        motion=SimpleNamespace(wait_client_ready=lambda *_args, **_kwargs: True),
        arm_controller_ready=lambda: True,
        _move_to_pregrasp_pose=lambda: True,
        _apply_gripper=lambda _width: True,
        _lock=nullcontext(),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    LlmControlTaskStateMachine(node).tick()

    assert node._state == LlmControlTaskState.IDLE.value


def test_pregrasp_waits_for_the_arm_controller_action_server():
    calls = []
    node = SimpleNamespace(
        _state=LlmControlTaskState.PREGRASP_POSE.value,
        abort=SimpleNamespace(is_set=lambda: False),
        motion=SimpleNamespace(wait_client_ready=lambda *_args, **_kwargs: True),
        arm_controller_ready=lambda: False,
        _move_to_pregrasp_pose=lambda: calls.append(True) or True,
        _apply_gripper=lambda _width: True,
        _lock=nullcontext(),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    LlmControlTaskStateMachine(node).tick()

    assert calls == []
    assert node._state == LlmControlTaskState.PREGRASP_POSE.value


def test_pregrasp_motion_is_single_flight_while_the_timer_ticks():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def move_to_pregrasp():
        calls.append(True)
        entered.set()
        assert release.wait(timeout=1.0)
        return True

    node = SimpleNamespace(
        _state=LlmControlTaskState.PREGRASP_POSE.value,
        abort=SimpleNamespace(is_set=lambda: False),
        motion=SimpleNamespace(wait_client_ready=lambda *_args, **_kwargs: True),
        arm_controller_ready=lambda: True,
        _move_to_pregrasp_pose=move_to_pregrasp,
        _apply_gripper=lambda _width: True,
        _lock=threading.Lock(),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )
    machine = LlmControlTaskStateMachine(node)
    first_tick = threading.Thread(target=machine.tick)

    first_tick.start()
    assert entered.wait(timeout=1.0)
    machine.tick()
    release.set()
    first_tick.join(timeout=1.0)

    assert calls == [True]
    assert node._state == LlmControlTaskState.IDLE.value


def test_pick_flow_uses_yolo_named_stages_and_returns_to_pregrasp_holding():
    calls = []
    source = object()
    node = SimpleNamespace(
        _state="IDLE",
        _lock=nullcontext(),
        open_finger_position=0.0305,
        _pick_preview_poses=lambda _source: {
            "approach_pick": "above", "grasp": "grasp", "carry": "lift",
        },
        pregrasp_pose="pregrasp",
        _apply_gripper=lambda width: calls.append(("gripper", width)) or True,
        _move_pose=lambda pose, name, *args: calls.append((name, pose, *args)) or True,
        _move_to_pregrasp_pose=lambda: calls.append(("pregrasp", "pregrasp")) or True,
        _execution_interrupted=lambda *_args: False,
        _mark_stop_state=lambda: None,
        _feedback=lambda *_args: None,
        _held_source=None,
    )

    result = LlmControlTaskStateMachine(node).execute_action(
        {"type": "pick", "source": source}, "goal", 0, 6, 1
    )

    assert result == (True, "pick complete; holding object", 6)
    assert node._state == LlmControlTaskState.HOLDING.value
    assert node._held_source is source
    moves = [call for call in calls if call[0] != "gripper"]
    assert [call[0] for call in moves] == [
        "approach_pick", "grasp", "carry", "pregrasp",
    ]
    assert moves[0][-1] == 0.2
    assert moves[1][-1] == 0.02
    assert moves[2][-1] == 0.2
    assert [call for call in calls if call[0] == "gripper"] == [
        ("gripper", 0.061), ("gripper", 0.0), ("gripper", 0.0),
    ]


def test_empty_place_executes_release_path_without_holding_recovery():
    calls = []
    node = SimpleNamespace(
        _state="IDLE",
        _lock=nullcontext(),
        open_finger_position=0.0305,
        _place_preview_poses=lambda source, _destination: (
            calls.append(("pose_source", source))
            or {"approach_box": "above", "release": "release"}
        ),
        pregrasp_pose="pregrasp",
        _apply_gripper=lambda width: calls.append(("gripper", width)) or True,
        _move_pose=lambda pose, name, *args: calls.append((name, pose, *args)) or True,
        _move_to_pregrasp_pose=lambda: calls.append(("pregrasp", "pregrasp")) or True,
        _execution_interrupted=lambda *_args: False,
        _mark_stop_state=lambda: None,
        _feedback=lambda *_args: None,
        _mark_holding_recovery=lambda *_args: calls.append(("holding_recovery",)),
        _clear_holding_locked=lambda: calls.append(("clear_holding",)),
        _held_source=None,
    )

    result = LlmControlTaskStateMachine(node).execute_action(
        {"type": "place", "source": None, "destination": object()}, "goal", 0, 4, 1
    )

    assert result == (True, "placed into previewed box", 4)
    assert calls == [
        ("pose_source", None),
        ("approach_box", "above", False, 0.2),
        ("release", "release", True, 0.2),
        ("gripper", 0.061),
        ("pregrasp", "pregrasp"),
        ("gripper", 0.0),
    ]


def test_unknown_action_never_falls_back_to_pregrasp_motion():
    node = SimpleNamespace()
    result = LlmControlTaskStateMachine(node).execute_action(
        {"type": "home"}, "goal", 0, 1, 1
    )
    assert result == (False, "unsupported action: home", 0)
