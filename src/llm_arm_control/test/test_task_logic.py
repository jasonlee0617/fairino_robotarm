import pytest

from llm_arm_control_nodes.agent_protocol import (
    AGENT_TOOLS,
    SYSTEM_PROMPT,
    agent_tool_definitions,
    parse_agent_tool_call,
    protocol_failure_response,
    scene_response_instructions,
)
from llm_arm_control_nodes.task_logic import (
    CapabilityInvocation,
    PreviewFailure,
    SceneEntity,
    spoken_error_text,
    task_plan_for_invocation,
)


def entity(index, class_name, u, v, *, selectable=True, distance=0.2):
    return SceneEntity(
        entity_id=f"e{index}",
        role="destination" if class_name == "box" else "pickable",
        class_name=class_name,
        confidence=0.9,
        image_u=u,
        image_v=v,
        image_width=1280.0,
        image_height=720.0,
        image_center_distance=((u - 0.5) ** 2 + (v - 0.5) ** 2) ** 0.5,
        obb_area_px=100.0,
        depth_m=0.6,
        depth_quality=0.9,
        base_x=0.2,
        base_y=0.3,
        base_z=0.02,
        tool_distance=distance,
        selectable=selectable,
        unavailable_reason="" if selectable else "VISION_DEPTH_INVALID",
        frame_stamp=1,
        result_seq=2,
        detection_index=index,
    )


SCENE = (
    entity(0, "elongated_object", 0.20, 0.50, distance=0.4),
    entity(1, "elongated_object", 0.51, 0.51, distance=0.3),
    entity(2, "elongated_object", 0.85, 0.45, distance=0.2),
    entity(3, "cube", 0.30, 0.20, distance=0.5),
    entity(4, "box", 0.70, 0.80),
)


def relative_step(x=0.0, y=0.0, z=0.0, rx=0.0, ry=0.0, rz=0.0):
    return {
        "x_m": x, "y_m": y, "z_m": z,
        "rx_deg": rx, "ry_deg": ry, "rz_deg": rz,
    }


def test_system_prompt_is_shorter_without_losing_safety_contract():
    assert len(SYSTEM_PROMPT) < 1160
    for required in (
        "CURRENT_YOLO_SCENE", "scene_id", "source_entity_ids", "base_link",
        "0.30m", "60度", "碰撞感知IK", "MoveIt", "不联网", "home",
    ):
        assert required in SYSTEM_PROMPT


def test_dynamic_scene_and_repair_prompts_are_compact_and_schema_driven():
    scene_prompt = scene_response_instructions({"scene_id": "scene-1", "entities": []})
    assert scene_prompt.startswith("CURRENT_YOLO_SCENE=")
    assert len(scene_prompt) < 120

    value, instructions = protocol_failure_response(
        "submit_visual_task", "MODEL_RESPONSE_INVALID", "missing scene_id",
        final=False,
    )
    assert value["expected_fields"] == (
        "operation, scene_id, source_entity_ids, destination_entity_id"
    )
    assert "scene_id" in instructions and "按schema重调" in instructions


def test_error_speech_uses_explicit_codes():
    assert spoken_error_text("TARGET_UNREACHABLE").startswith("目标未通过")
    assert spoken_error_text("PLAN_INVALID").startswith("任务计划未通过")
    assert "故障代码" in spoken_error_text("UNEXPECTED_CODE")


def test_entity_id_selection_compiles_to_local_indices():
    invocation = CapabilityInvocation(
        "yolo.pick_place", "scene-1", ("e0", "e1", "e2"), "e4"
    )
    plan = task_plan_for_invocation(invocation, SCENE[:3], [SCENE[4]])
    assert [action["source_index"] for action in plan.actions] == [0, 1, 2]
    assert all(action["destination_index"] == 4 for action in plan.actions)


def test_unavailable_detected_box_preserves_specific_visual_error():
    unavailable_box = entity(4, "box", 0.7, 0.8, selectable=False)
    invocation = CapabilityInvocation("yolo.pick_place", "scene-1", ("e0",), "e4")

    with pytest.raises(PreviewFailure) as failure:
        task_plan_for_invocation(invocation, [SCENE[0]], [unavailable_box])

    assert failure.value.error_code == "VISION_DEPTH_INVALID"


@pytest.mark.parametrize(
    "step,error",
    (
        (relative_step(x=0.3001), "0.30"),
        (relative_step(y=-0.3001), "0.30"),
        (relative_step(z=0.3001), "0.30"),
        (relative_step(rx=60.1), "60"),
        (relative_step(ry=-60.1), "60"),
        (relative_step(rz=60.1), "60"),
    ),
)
def test_relative_motion_is_bounded_per_axis(step, error):
    with pytest.raises(ValueError, match=error):
        parse_agent_tool_call(
            "move_relative",
            {"steps": [step]},
            allowed_tools=("move_relative",),
        )


def test_relative_motion_supports_combined_six_axis_base_delta():
    _name, invocation = parse_agent_tool_call(
        "move_relative",
        {"steps": [
            relative_step(0.30, -0.30, 0.30, 60.0, -60.0, 30.0),
            relative_step(y=0.05),
        ]},
        allowed_tools=("move_relative",),
    )
    actions = task_plan_for_invocation(invocation).actions

    assert actions[0] == {
        "type": "move_relative",
        "dx": 0.30, "dy": -0.30, "dz": 0.30,
        "droll_deg": 60.0, "dpitch_deg": -60.0, "dyaw_deg": 30.0,
        "frame_id": "base_link",
    }
    assert actions[1]["dy"] == 0.05
    assert all(item["frame_id"] == "base_link" for item in actions)


@pytest.mark.parametrize(
    "step",
    (
        relative_step(),
        relative_step(x=float("nan")),
        relative_step(rx=float("inf")),
    ),
)
def test_relative_motion_rejects_zero_and_nonfinite_steps(step):
    with pytest.raises(ValueError):
        parse_agent_tool_call(
            "move_relative", {"steps": [step]},
            allowed_tools=("move_relative",),
        )


def test_relative_motion_rejects_old_direction_protocol():
    with pytest.raises(ValueError, match="unknown fields"):
        parse_agent_tool_call(
            "move_relative",
            {"steps": [{"direction": "left", "distance_m": 0.1}]},
            allowed_tools=("move_relative",),
        )


def test_tool_allowlist_and_schema_are_enforced_locally():
    definitions = agent_tool_definitions(("set_gripper", "ask_user"))
    assert [item["function"]["name"] for item in definitions] == ["set_gripper", "ask_user"]
    with pytest.raises(ValueError, match="not allowed"):
        parse_agent_tool_call("set_gripper", {"state": "open"}, allowed_tools=("ask_user",))
    with pytest.raises(ValueError, match="unknown fields: extra"):
        parse_agent_tool_call(
            "set_gripper", {"state": "open", "extra": 1},
            allowed_tools=("set_gripper",),
        )


def test_shallow_gripper_and_visual_tools_compile_to_existing_domain_types():
    name, gripper = parse_agent_tool_call(
        "set_gripper", {"state": "close"}, allowed_tools=("set_gripper",)
    )
    assert name == "commit_task"
    assert task_plan_for_invocation(gripper).actions == (
        {"type": "set_gripper", "state": "close"},
    )

    name, visual = parse_agent_tool_call(
        "submit_visual_task",
        {
            "operation": "pick_place",
            "scene_id": "scene-1",
            "source_entity_ids": ["e0", "e1"],
            "destination_entity_id": "e4",
        },
        allowed_tools=("submit_visual_task",),
    )
    assert name == "commit_task"
    assert visual.skill == "yolo.pick_place"
    assert visual.source_entity_ids == ("e0", "e1")


@pytest.mark.parametrize(
    ("operation", "source", "destination"),
    (
        ("pick", ["e0"], ""),
        ("place", [], "e4"),
        ("pick_place", ["e0"], "e4"),
    ),
)
def test_visual_operation_requires_the_matching_branch_groups(
    operation, source, destination,
):
    _name, invocation = parse_agent_tool_call(
        "submit_visual_task",
        {
            "operation": operation,
            "scene_id": "scene-1", "source_entity_ids": source,
            "destination_entity_id": destination,
        },
        allowed_tools=("submit_visual_task",),
    )
    assert invocation.skill == "yolo." + operation


def test_multi_target_pick_requires_destination():
    with pytest.raises(PreviewFailure) as failure:
        parse_agent_tool_call(
            "submit_visual_task",
            {
                "operation": "pick", "scene_id": "scene-1",
                "source_entity_ids": ["e0", "e1"], "destination_entity_id": "",
            },
            allowed_tools=("submit_visual_task",),
        )
    assert failure.value.error_code == "DESTINATION_REQUIRED"


def test_old_queries_visual_protocol_is_rejected():
    with pytest.raises(ValueError, match="unknown fields: queries"):
        parse_agent_tool_call(
            "submit_visual_task",
            {"operation": "pick", "queries": []},
            allowed_tools=("submit_visual_task",),
        )

    with pytest.raises(ValueError, match="unknown fields"):
        parse_agent_tool_call(
            "submit_visual_task",
            {
                "operation": "pick",
                "source_branches": [], "destination_branches": [],
            },
            allowed_tools=("submit_visual_task",),
        )


def test_single_source_id_string_is_safely_normalized():
    _name, invocation = parse_agent_tool_call(
        "submit_visual_task",
        {
            "operation": "pick", "scene_id": "scene-1",
            "source_entity_ids": "e0", "destination_entity_id": "",
        },
        allowed_tools=("submit_visual_task",),
    )
    assert invocation.source_entity_ids == ("e0",)

    for invalid in ("", "e0,e1", "e0 e1", {}, 7):
        with pytest.raises(ValueError, match="source_entity_ids"):
            parse_agent_tool_call(
                "submit_visual_task",
                {
                    "operation": "pick", "scene_id": "scene-1",
                    "source_entity_ids": invalid, "destination_entity_id": "",
                },
                allowed_tools=("submit_visual_task",),
            )


def test_inspect_scene_boolean_is_strict():
    with pytest.raises(ValueError, match="must be a boolean"):
        parse_agent_tool_call(
            "inspect_scene", {"include_rgb": "false"},
            allowed_tools=("inspect_scene",),
        )


def test_capability_catalog_is_yolo_only_for_visual_tasks():
    schemas = agent_tool_definitions(("submit_visual_task", "set_gripper"))
    serialized = str(schemas)
    assert "graspnet.best_grasp" not in serialized
    assert "system.switch_mode" not in serialized
    assert "anyOf" not in serialized
    submit = schemas[0]["function"]["parameters"]
    assert submit["additionalProperties"] is False
    assert submit["properties"]["operation"]["enum"] == ["pick", "pick_place", "place"]
    assert submit["properties"]["source_entity_ids"]["uniqueItems"] is True
    assert "mode" not in submit["properties"]


def test_shallow_tool_catalog_is_less_than_half_the_old_size():
    import json

    encoded = json.dumps(agent_tool_definitions(AGENT_TOOLS), ensure_ascii=False)
    assert len(encoded) < 8304
    assert '"queries"' not in encoded
    assert '"source_branches"' not in encoded
    assert '"destination_branches"' not in encoded
    assert encoded.count('"source_entity_ids"') == 2
