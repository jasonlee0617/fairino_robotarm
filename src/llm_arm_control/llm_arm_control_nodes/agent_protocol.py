"""Qwen-facing tool schemas, prompts, and strict argument parsing."""

from __future__ import annotations

import json
import math
import re

from llm_arm_control_nodes.task_logic import (
    CapabilityInvocation,
    MAX_ACTIONS,
    PreviewFailure,
    RELATIVE_STEP_FIELDS,
    VISUAL_ACTIONS,
    validate_spoken_text,
)


def _relative_step_schema():
    properties = {}
    for name in RELATIVE_STEP_FIELDS:
        bound = 0.30 if name.endswith("_m") else 60.0
        properties[name] = {
            "type": "number", "minimum": -bound, "maximum": bound,
        }
    return {
        "type": "object",
        "required": list(RELATIVE_STEP_FIELDS),
        "additionalProperties": False,
        "properties": properties,
    }


AGENT_TOOL_SPECS = {
    "submit_visual_task": (
        "Submit selected IDs from CURRENT_YOLO_SCENE. Source ID order is execution order.",
        {
            "type": "object",
            "required": [
                "operation", "scene_id", "source_entity_ids",
                "destination_entity_id",
            ],
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string", "enum": sorted(VISUAL_ACTIONS),
                },
                "scene_id": {"type": "string", "minLength": 1},
                "source_entity_ids": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": MAX_ACTIONS,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "destination_entity_id": {"type": "string"},
            },
        },
    ),
    "move_relative": (
        "Move relative to base_link using ordered six-axis delta steps. Positive x/y/z "
        "mean forward/left/up; rotations use fixed base axes and the right-hand rule.",
        {
            "type": "object",
            "required": ["steps"],
            "additionalProperties": False,
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_ACTIONS,
                    "items": _relative_step_schema(),
                },
            },
        },
    ),
    "set_gripper": (
        "Open or close the gripper.",
        {
            "type": "object",
            "required": ["state"],
            "additionalProperties": False,
            "properties": {
                "state": {"type": "string", "enum": ["open", "close"]},
            },
        },
    ),
    "ask_user": (
        "Ask for one genuinely missing preference or unsupported attribute.",
        {
            "type": "object",
            "required": ["question", "missing_field"],
            "additionalProperties": False,
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "missing_field": {
                    "type": "string",
                    "enum": [
                        "target", "destination", "order",
                        "unsupported_attribute", "other",
                    ],
                },
            },
        },
    ),
    "cancel_task": (
        "Cancel the unfinished task.",
        {
            "type": "object",
            "required": ["summary"],
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string", "minLength": 1},
            },
        },
    ),
    "inspect_scene": (
        "Read the current YOLO scene. Use include_rgb only for visual details absent "
        "from structured entities.",
        {
            "type": "object",
            "required": ["include_rgb"],
            "additionalProperties": False,
            "properties": {"include_rgb": {"type": "boolean"}},
        },
    ),
}
AGENT_TOOLS = tuple(AGENT_TOOL_SPECS)
AGENT_TOOL_FIELDS = {
    name: tuple(spec[1]["required"])
    for name, spec in AGENT_TOOL_SPECS.items()
}

SYSTEM_PROMPT = """你是Fairino机械臂的高层任务决策器。机器人动作必须调用提供的闭集工具；普通非实时问答可直接回答，但当前会话不联网，时效性信息必须说明无法实时核验。
每轮会提供CURRENT_YOLO_SCENE。涉及当前画面、桌面目标或YOLO检测时，只能依据实体已观测字段回答，不得猜测颜色、材质、身份或未知类别；结构化字段不足以判断外观时才调用inspect_scene(include_rgb=true)。
类别：elongated_object为螺栓、螺丝、笔等细长物；cube为方块；box为role=destination的盒子。
submit_visual_task必须使用当前scene_id和entity_id，不得自造ID或使用检测索引。source_entity_ids顺序即执行顺序：pick恰有一个来源且目的地为空；place来源为空且指定一个盒子；pick_place有1至10个来源且指定一个盒子。依据class_name、role、image_u、image_v、image_center_distance、confidence、tool_distance理解方位、最近、全部和排除条件；多个目标必须分别提交，未指定顺序时按场景属性合理排序。工具调用前不得声称动作已开始或完成。
相对移动只用move_relative，夹爪只用set_gripper；信息不足用ask_user，取消用cancel_task。不得生成关节轨迹、绝对位姿，或调用、声称执行语音停止、复位、解锁、home等安全控制。
相对移动固定base_link：前后左右上下对应正负X/Y/Z。steps按用户顺序，每步必须给出x_m、y_m、z_m、rx_deg、ry_deg、rz_deg六个数值，未使用轴为0；每个平移轴绝对值不超过0.30m，每个旋转轴不超过60度，且至少一项非零。
YOLO抓放仍须通过本地重绑定、工作空间、碰撞感知IK和MoveIt验证，最多10个动作；关键位姿IK不代表完整轨迹成功。
摘要须简短可播报，不含Markdown、emoji、唤醒词或停止、复位、恢复口令。"""


def agent_tool_definitions(allowed_tools):
    return [
        {"type": "function", "function": {
            "name": name,
            "description": AGENT_TOOL_SPECS[name][0],
            "parameters": AGENT_TOOL_SPECS[name][1],
        }}
        for name in allowed_tools
    ]


def _finite_number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _clean_text(value, field):
    text = validate_spoken_text(value)
    if len(text) > 500:
        raise ValueError(f"{field} is too long")
    return text


def _require_fields(value, required, path):
    actual = set(value)
    unknown = sorted(actual - required)
    missing = sorted(required - actual)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(missing)}")


def _normalize_relative_steps(steps):
    if not isinstance(steps, list):
        raise ValueError("arm.move_relative requires an ordered steps array")
    if not 1 <= len(steps) <= MAX_ACTIONS:
        raise ValueError("relative motion requires 1 to 10 steps")
    normalized = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"relative motion step {index} must be an object")
        _require_fields(step, set(RELATIVE_STEP_FIELDS), f"relative motion step {index}")
        values = {
            field: _finite_number(step[field], f"steps[{index}].{field}")
            for field in RELATIVE_STEP_FIELDS
        }
        if any(abs(values[field]) > 0.30 for field in ("x_m", "y_m", "z_m")):
            raise ValueError("relative translation axis exceeds 0.30 m")
        if any(abs(values[field]) > 60.0 for field in ("rx_deg", "ry_deg", "rz_deg")):
            raise ValueError("relative rotation axis exceeds 60 degrees")
        if not any(values.values()):
            raise ValueError("relative motion step must contain a non-zero component")
        normalized.append({
            "type": "move_relative",
            "dx": values["x_m"], "dy": values["y_m"], "dz": values["z_m"],
            "droll_deg": values["rx_deg"],
            "dpitch_deg": values["ry_deg"],
            "dyaw_deg": values["rz_deg"],
            "frame_id": "base_link",
        })
    return tuple(normalized)


def parse_agent_tool_call(name: str, arguments: dict, *, allowed_tools):
    name = str(name)
    if name not in set(allowed_tools):
        raise ValueError(f"tool {name!r} is not allowed in the current state")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if name == "submit_visual_task":
        _require_fields(
            arguments,
            {"operation", "scene_id", "source_entity_ids", "destination_entity_id"},
            name,
        )
        operation = arguments["operation"]
        if not isinstance(operation, str) or operation not in VISUAL_ACTIONS:
            raise ValueError("submit_visual_task.operation is unsupported")
        if not isinstance(arguments["scene_id"], str):
            raise ValueError("submit_visual_task.scene_id must be a string")
        scene_id = arguments["scene_id"].strip()
        source_ids = arguments["source_entity_ids"]
        if isinstance(source_ids, str) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", source_ids.strip()
        ):
            source_ids = [source_ids.strip()]
        if not isinstance(arguments["destination_entity_id"], str):
            raise ValueError("submit_visual_task.destination_entity_id must be a string")
        destination_id = arguments["destination_entity_id"].strip()
        if not scene_id:
            raise ValueError("submit_visual_task.scene_id must not be empty")
        if (
            not isinstance(source_ids, list)
            or len(source_ids) > MAX_ACTIONS
            or any(not isinstance(item, str) or not item.strip() for item in source_ids)
        ):
            raise ValueError("submit_visual_task.source_entity_ids must contain 0 to 10 strings")
        source_ids = tuple(item.strip() for item in source_ids)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("submit_visual_task.source_entity_ids contains duplicates")
        if operation == "pick":
            if len(source_ids) > 1:
                raise PreviewFailure(
                    "DESTINATION_REQUIRED",
                    "multiple pick targets require a destination",
                )
            if len(source_ids) != 1 or destination_id:
                raise ValueError("pick requires one source and an empty destination")
        elif operation == "place":
            if source_ids or not destination_id:
                raise ValueError("place requires no source and one destination")
        elif not source_ids or not destination_id:
            raise ValueError("pick_place requires 1 to 10 sources and one destination")
        return "commit_task", CapabilityInvocation(
            skill="yolo." + operation,
            scene_id=scene_id,
            source_entity_ids=source_ids,
            destination_entity_id=destination_id,
        )
    if name == "move_relative":
        _require_fields(arguments, {"steps"}, name)
        return "commit_task", CapabilityInvocation(
            "arm.move_relative",
            parameters={"steps": _normalize_relative_steps(arguments["steps"])},
        )
    if name == "set_gripper":
        _require_fields(arguments, {"state"}, name)
        if arguments["state"] not in ("open", "close"):
            raise ValueError("set_gripper.state must be open or close")
        return "commit_task", CapabilityInvocation(
            "gripper.set", parameters={"state": arguments["state"]},
        )
    if name == "ask_user":
        _require_fields(arguments, {"question", "missing_field"}, name)
        missing = str(arguments["missing_field"])
        if missing not in (
            "target", "destination", "order", "unsupported_attribute", "other"
        ):
            raise ValueError("ask_user missing_field is unsupported")
        return name, {
            "question": _clean_text(arguments["question"], "question"),
            "missing_field": missing,
        }
    if name == "cancel_task":
        if set(arguments) != {"summary"}:
            raise ValueError("cancel_task requires only summary")
        return name, _clean_text(arguments["summary"], "summary")
    if name == "inspect_scene":
        _require_fields(arguments, {"include_rgb"}, name)
        if not isinstance(arguments["include_rgb"], bool):
            raise ValueError("inspect_scene.include_rgb must be a boolean")
        return name, {"include_rgb": arguments["include_rgb"]}
    raise ValueError("unsupported agent tool")


def normalize_tool_arguments(value) -> str:
    parsed = value if isinstance(value, dict) else json.loads(str(value or "{}"))
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def scene_response_instructions(scene) -> str:
    compact = json.dumps(scene, ensure_ascii=False, separators=(",", ":"))
    return "CURRENT_YOLO_SCENE=" + compact + "\n按系统场景规则回答或调用工具。"


def tool_argument_shape(name: str, arguments: dict) -> dict:
    if name != "submit_visual_task":
        return {}
    source_ids = arguments.get("source_entity_ids")
    return {
        "scene_id_present": bool(arguments.get("scene_id")),
        "source_ids_type": (
            "array" if isinstance(source_ids, list) else type(source_ids).__name__
        ),
        "source_ids_count": len(source_ids) if isinstance(source_ids, list) else 0,
        "destination_id_present": bool(arguments.get("destination_entity_id")),
    }


def spoken_only_instructions(text: str) -> str:
    return f"只朗读以下内容，不添加或改写：{text}"


def protocol_failure_response(name, error_code, detail, *, final):
    spoken = "模型生成的任务参数不完整，本次任务未执行，请重新描述。"
    expected = ", ".join(AGENT_TOOL_FIELDS.get(name, ())) or "当前工具定义"
    value = {
        "accepted": False,
        "status": "rejected",
        "error_code": error_code,
        "detail": detail,
        "expected_fields": expected,
        **({"spoken_text": spoken} if final else {}),
    }
    if final:
        return value, spoken_only_instructions(spoken)
    hint = (
        "scene_id取自CURRENT_YOLO_SCENE；source_entity_ids按执行顺序；"
        "destination_entity_id为盒子ID，pick时为空。"
        if name == "submit_visual_task" else ""
    )
    instructions = (
        f"{name}调用无效：{detail}。必填字段：{expected}。{hint}"
        "立即按schema重调同一工具，不要先说话。"
    )
    return value, instructions


def compact_tool_result(name, result):
    value = {
        "accepted": bool(result.get("accepted")),
        "status": str(result.get("status", "rejected")),
        "error_code": str(result.get("error_code", "")),
    }
    for field in ("spoken_text", "detail"):
        if result.get(field):
            value[field] = result[field]
    if name == "inspect_scene":
        for field in (
            "entities", "diagnostics", "scene_error", "include_rgb",
            "scene_id", "rgb_attached", "rgb_error",
        ):
            if field in result:
                value[field] = result[field]
    return value
