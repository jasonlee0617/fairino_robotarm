"""Pure domain rules for bounded LLM robot tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import time
from typing import Iterable


PICK_EXECUTION_STEPS = 6
PLACE_EXECUTION_STEPS = 4
PICK_PLACE_EXECUTION_STEPS = 10
VISUAL_ACTIONS = frozenset({"pick", "place", "pick_place"})
MAX_ACTIONS = 10
RELATIVE_STEP_FIELDS = (
    "x_m", "y_m", "z_m", "rx_deg", "ry_deg", "rz_deg",
)


@dataclass(frozen=True)
class SceneEntity:
    entity_id: str
    role: str
    class_name: str
    confidence: float
    image_u: float
    image_v: float
    image_width: float
    image_height: float
    image_center_distance: float
    obb_area_px: float
    depth_m: float | None
    depth_quality: float
    base_x: float | None
    base_y: float | None
    base_z: float | None
    tool_distance: float | None
    selectable: bool
    unavailable_reason: str
    frame_stamp: int
    result_seq: int
    detection_index: int = field(repr=False)

    def public(self):
        value = asdict(self)
        value.pop("detection_index", None)
        return value


@dataclass(frozen=True)
class CapabilityInvocation:
    skill: str
    scene_id: str = ""
    source_entity_ids: tuple[str, ...] = ()
    destination_entity_id: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskPlan:
    actions: tuple[dict, ...]


@dataclass
class TaskContext:
    scene_id: str = ""
    scene_created_at: float = 0.0
    entities: tuple[SceneEntity, ...] = ()


@dataclass(frozen=True)
class TaskPreview:
    preview_id: str
    plan: TaskPlan
    created_at: float
    max_age_sec: float = 15.0


@dataclass
class PreviewRecord:
    preview: TaskPreview
    session_id: str
    enriched_actions: list[dict]
    safety_epoch: int


class PreviewFailure(ValueError):
    def __init__(self, error_code: str, detail: str):
        super().__init__(detail)
        self.error_code = str(error_code)


@dataclass(frozen=True)
class SafetyState:
    epoch: int = 0
    blocked: bool = False
    command: str = ""


def task_plan_for_invocation(invocation, sources=(), destinations=()):
    skill = invocation.skill
    if skill == "arm.move_relative":
        return TaskPlan(tuple(dict(step) for step in invocation.parameters["steps"]))
    if skill == "gripper.set":
        return TaskPlan(({"type": "set_gripper", "state": invocation.parameters["state"]},))
    sources, destinations = list(sources), list(destinations)
    if skill in ("yolo.pick", "yolo.pick_place"):
        if not sources:
            raise PreviewFailure("TARGET_NOT_FOUND", "no requested target was detected")
        if any(not source.selectable for source in sources):
            unavailable = next(source for source in sources if not source.selectable)
            raise PreviewFailure(
                unavailable.unavailable_reason or "VISION_NO_SELECTABLE_TARGET",
                "requested targets include unavailable objects",
            )
        if len(sources) > MAX_ACTIONS:
            raise PreviewFailure(
                "ACTION_LIMIT_EXCEEDED", "one task can contain at most 10 targets"
            )
    if skill in ("yolo.place", "yolo.pick_place"):
        selectable = [destination for destination in destinations if destination.selectable]
        if not selectable:
            unavailable = {
                destination.unavailable_reason
                for destination in destinations
                if destination.unavailable_reason
            }
            for code in (
                "VISION_TF_UNAVAILABLE",
                "VISION_BOX_NO_FREE_REGION",
                "VISION_DEPTH_INVALID",
                "VISION_OBB_AXIS_INVALID",
            ):
                if code in unavailable:
                    raise PreviewFailure(
                        code,
                        f"box detected in 2D but unavailable: {code}",
                    )
            raise PreviewFailure(
                "TARGET_NOT_FOUND", "no selectable box destination was detected"
            )
        if len(selectable) > 1:
            raise PreviewFailure(
                "DESTINATION_REQUIRED", "multiple destinations require clarification"
            )
        destination = selectable[0]
    if skill == "yolo.pick":
        if len(sources) != 1:
            raise PreviewFailure(
                "MODEL_RESPONSE_INVALID", "yolo.pick requires exactly one target"
            )
        actions = ({"type": "pick", "source_index": sources[0].detection_index},)
    elif skill == "yolo.place":
        actions = ({"type": "place", "destination_index": destination.detection_index},)
    else:
        actions = tuple({
            "type": "pick_place",
            "source_index": source.detection_index,
            "destination_index": destination.detection_index,
        } for source in sources)
    return TaskPlan(actions)


def validate_visual_state(action_type: str, *, holding: bool, recovery: bool) -> None:
    if action_type not in VISUAL_ACTIONS:
        return
    if recovery:
        raise PreviewFailure(
            "SAFETY_BLOCKED",
            "placement recovery must be retried or cleared with keyboard h",
        )
    if action_type != "place" and holding:
        raise PreviewFailure(
            "ARM_HOLDING",
            "the arm is already holding an object; place it or press h first",
        )


_SPOKEN_RESERVED = (
    "小鹏同学", "小鹏小鹏", "hirobot", "小鹏结束控制", "robotendsession",
    "急停", "停止", "stop", "复位", "reset", "home", "回到安全位",
    "恢复控制", "继续控制", "resume", "解锁", "unlock",
)
_MARKDOWN = re.compile(r"[`#*_\[\]<>]")
_EMOJI = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]")


def validate_spoken_text(text) -> str:
    value = sanitize_spoken_text(text)
    if not value:
        raise ValueError("spoken text must not be empty after sanitizing")
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())
    if any(token in compact for token in _SPOKEN_RESERVED):
        raise ValueError("spoken text contains a reserved local control phrase")
    return value


def sanitize_spoken_text(text) -> str:
    """Clean trusted local speech without applying model-output policy."""
    value = re.sub(r"\s+", " ", str(text)).strip()
    value = re.sub(r"\s+", " ", _EMOJI.sub("", _MARKDOWN.sub("", value))).strip()
    if not value:
        raise ValueError("spoken text must not be empty after sanitizing")
    return value


_ZH_DIGITS = "零一二三四五六七八九"


def chinese_number(value: float) -> str:
    """Render bounded motion values deterministically for Chinese speech."""
    number = float(value)
    sign = "负" if number < 0 else ""
    text = f"{abs(number):.6f}".rstrip("0").rstrip(".")
    integer, dot, fraction = text.partition(".")
    n = int(integer)
    if n < 10:
        head = _ZH_DIGITS[n]
    elif n < 20:
        head = "十" + ("" if n == 10 else _ZH_DIGITS[n % 10])
    elif n < 100:
        head = _ZH_DIGITS[n // 10] + "十" + ("" if n % 10 == 0 else _ZH_DIGITS[n % 10])
    else:
        head = "".join(_ZH_DIGITS[int(digit)] for digit in integer)
    tail = "" if not dot else "点" + "".join(_ZH_DIGITS[int(digit)] for digit in fraction)
    return sign + head + tail


def invocation_spoken_text(
    invocation: CapabilityInvocation, action_count: int | None = None
) -> str:
    """Build the safety-relevant start sentence from validated local parameters."""
    if invocation.skill == "yolo.pick_place":
        return f"准备执行{chinese_number(action_count or 0)}次抓放任务。"
    if invocation.skill == "yolo.pick":
        return f"准备抓取{chinese_number(action_count or 0)}个目标。"
    if invocation.skill == "yolo.place":
        return "准备放置当前夹持物体。"
    if invocation.skill == "gripper.set":
        return "准备张开夹爪。" if invocation.parameters["state"] == "open" else "准备闭合夹爪。"
    phrases = []
    translations = (
        ("dx", "向前", "向后"), ("dy", "向左", "向右"),
        ("dz", "向上", "向下"),
    )
    rotations = (
        ("droll_deg", "X"), ("dpitch_deg", "Y"), ("dyaw_deg", "Z"),
    )
    for step in invocation.parameters["steps"]:
        components = []
        for key, positive, negative in translations:
            value = step[key]
            if value:
                direction = positive if value > 0.0 else negative
                components.append(
                    f"{direction}移动{chinese_number(abs(value) * 100.0)}厘米"
                )
        for key, axis in rotations:
            value = step[key]
            if value:
                direction = "正向" if value > 0.0 else "负向"
                components.append(
                    f"绕基座{axis}轴{direction}旋转{chinese_number(abs(value))}度"
                )
        phrases.append("，同时".join(components))
    return "机械臂" + "，然后".join(phrases) + "。"


def preview_status(preview: TaskPreview, now=None) -> str:
    now = time.monotonic() if now is None else float(now)
    age = now - preview.created_at
    return "ready" if 0.0 <= age <= preview.max_age_sec else "expired"


def apply_safety_command(state: SafetyState, command: str) -> SafetyState:
    command = str(command).strip().lower()
    if command in ("stop", "reset"):
        return SafetyState(state.epoch + 1, True, command)
    if command == "resume":
        return SafetyState(state.epoch, False, command)
    return state


def complete_safety_reset(state: SafetyState) -> SafetyState:
    return SafetyState(state.epoch, False, "")


def safety_execution_valid(state: SafetyState, execution_epoch: int) -> bool:
    return not state.blocked and state.epoch == int(execution_epoch)


_ERROR_SPEECH = {
    "ROBOT_BUSY": "机械臂正在执行任务，本次指令未排队。",
    "ROBOT_INITIALIZING": "机械臂正在初始化，请稍后重新下达任务。",
    "SAFETY_BLOCKED": "机械臂处于安全锁定状态，请检查现场后按键盘h复位。",
    "VISION_UNAVAILABLE": "视觉数据暂时不可用，请检查相机数据流后重试。",
    "VISION_DEPTH_STALE": "深度图像已过期，请检查深度相机数据流。",
    "VISION_SYNC_STALE": "彩色图和深度图不同步，请检查相机与桥接。",
    "VISION_DEPTH_INVALID": "目标区域没有有效深度，当前不能安全执行。",
    "VISION_TF_UNAVAILABLE": "相机到机械臂基座的坐标变换不可用。",
    "VISION_OBB_AXIS_INVALID": "目标方向检测无效，当前不能安全执行。",
    "VISION_BOX_NO_FREE_REGION": "盒内没有安全放置空间。",
    "VISION_NO_SELECTABLE_TARGET": "检测到目标，但当前目标不可安全执行。",
    "TARGET_NOT_FOUND": "没有检测到符合要求且可执行的目标。",
    "BATCH_SOURCE_NOT_STABLE": "剩余目标未能连续稳定识别，当前批量任务已终止。",
    "BATCH_DESTINATION_NOT_VISIBLE": "没有实时检测到盒子，当前批量任务已终止。",
    "SCENE_EXPIRED": "当前场景已更新，请重新说出任务。",
    "DESTINATION_REQUIRED": "多个抓取目标需要说明放置位置。",
    "TARGET_UNREACHABLE": "目标未通过工作空间或碰撞感知逆解检查。",
    "ACTION_LIMIT_EXCEEDED": "单次任务最多执行十个动作，请拆分任务。",
    "MODEL_RESPONSE_INVALID": "智能体返回的工具参数不符合协议，请重新表达任务。",
    "PLAN_INVALID": "任务计划未通过本地校验，本次任务未执行。",
    "ARM_HOLDING": "机械臂已经夹持物体，请先放置物体或按键盘h复位。",
}


def spoken_error_text(error_code: str) -> str:
    return _ERROR_SPEECH.get(
        str(error_code), "任务无法执行，请查看故障代码后重试。"
    )


def execution_step_count(actions: TaskPlan | Iterable[dict]) -> int:
    actions = actions.actions if isinstance(actions, TaskPlan) else actions
    counts = {
        "pick": PICK_EXECUTION_STEPS,
        "place": PLACE_EXECUTION_STEPS,
        "pick_place": PICK_PLACE_EXECUTION_STEPS,
    }
    return sum(counts.get(action.get("type"), 1) for action in actions)
