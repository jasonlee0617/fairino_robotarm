from myrobot_common.nodes.motion_control_node import (
    motion_command_for_key,
    trajectory_event_for_command,
)


def test_stop_and_reset_map_to_moveit_stop_event():
    assert trajectory_event_for_command("stop") == "stop"
    assert trajectory_event_for_command("reset") == "stop"


def test_resume_and_unknown_commands_do_not_publish_moveit_events():
    assert trajectory_event_for_command("resume") is None
    assert trajectory_event_for_command("g") is None
    assert trajectory_event_for_command("unknown") is None


def test_interactive_safety_keys_share_one_command_mapping():
    assert motion_command_for_key(" ") == "stop"
    assert motion_command_for_key("H") == "reset"
    assert motion_command_for_key("r") == "resume"
    assert motion_command_for_key("g") == "g"
