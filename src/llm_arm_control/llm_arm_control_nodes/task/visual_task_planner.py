"""Visual target conversion, rebinding, and deterministic preview planning."""

from __future__ import annotations

import json
import math
import time
import uuid

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from llm_arm_control_nodes.task.llm_control_state_machine import LlmControlTaskState
from llm_arm_control_nodes.task_logic import (
    CapabilityInvocation,
    PreviewFailure,
    PreviewRecord,
    SceneEntity,
    TaskPlan,
    TaskPreview,
    safety_execution_valid,
    task_plan_for_invocation,
    validate_visual_state,
)
from visual_perception_utils.llm_rgbd import ResolvedCandidate


def resolved_candidate(item) -> ResolvedCandidate:
    return ResolvedCandidate(
        index=int(item["index"]),
        class_name=str(item["class_name"]),
        confidence=float(item.get("confidence", 0.0)),
        center_uv=tuple(float(value) for value in item.get("center_uv", (0.0, 0.0))),
        xyz=tuple(float(value) for value in item["base_xyz"]),
        yaw=float(item.get("yaw", 0.0)),
        frame_stamp_ns=int(item.get("frame_stamp_ns", 0)),
        depth_inlier_ratio=float(item.get("depth_inlier_ratio", 0.0)),
        result_seq=int(item.get("result_seq", 0)),
        depth_m=float(item.get("depth_m", 0.0)),
        obb_area_px=float(item.get("obb_area_px", 0.0)),
        placement_uv=(
            None if item.get("placement_uv") is None
            else tuple(float(value) for value in item["placement_uv"])
        ),
        container_xyz=(
            None if item.get("container_base_xyz") is None
            else tuple(float(value) for value in item["container_base_xyz"])
        ),
    )


def match_batch_targets(frozen, current, max_distance_m):
    required = {}
    available = {}
    for item in frozen:
        required[item.class_name] = required.get(item.class_name, 0) + 1
    for item in current:
        available[item.class_name] = available.get(item.class_name, 0) + 1
    if any(available.get(name, 0) < count for name, count in required.items()):
        return None
    costs = [
        [
            math.hypot(old.xyz[0] - new.xyz[0], old.xyz[1] - new.xyz[1])
            if old.class_name == new.class_name else float("inf")
            for new in current
        ]
        for old in frozen
    ]
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError:
        return None
    assignments = {row: column for row, column in zip(rows, columns)}
    if len(assignments) != len(frozen) or any(
        not math.isfinite(costs[row][column])
        or costs[row][column] > float(max_distance_m)
        for row, column in assignments.items()
    ):
        return None
    return assignments, costs


class VisualTaskPlannerMixin:
    """Own visual target resolution, rebinding, and preview compilation."""

    def _wait_for_planning_metadata(self, *, after_result_seq=None):
        if after_result_seq is None:
            return self.perception.wait_for_planning_metadata()
        return self.perception.wait_for_planning_metadata(
            after_result_seq=after_result_seq
        )

    def _current_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, self.ee_frame, Time(), timeout=Duration(seconds=0.2)
        )
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def _scene_entities(self, metadata, current_pose=None):
        if current_pose is None:
            try:
                current_pose = self._current_pose()
            except Exception:
                current_pose = None
        current_xyz = None if current_pose is None else (
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
        )
        entities = []
        for ordinal, item in enumerate(metadata, start=1):
            center = item.get("center_uv") or (0.0, 0.0)
            image_size = item.get("image_size") or (1.0, 1.0)
            width, height = max(float(image_size[0]), 1.0), max(float(image_size[1]), 1.0)
            xyz = (
                item.get("container_base_xyz") or item.get("base_xyz")
                if item.get("class_name") in self.place_classes
                else item.get("base_xyz")
            )
            entities.append(SceneEntity(
                entity_id=f"entity-{ordinal}",
                role=(
                    "destination"
                    if item.get("class_name") in self.place_classes
                    else "pickable"
                ),
                class_name=str(item.get("class_name", "")),
                confidence=float(item.get("confidence", 0.0)),
                image_u=float(center[0]) / width,
                image_v=float(center[1]) / height,
                image_width=width,
                image_height=height,
                image_center_distance=math.hypot(
                    float(center[0]) / width - 0.5,
                    float(center[1]) / height - 0.5,
                ),
                obb_area_px=float(item.get("obb_area_px", 0.0)),
                depth_m=(None if item.get("depth_m") is None else float(item["depth_m"])),
                depth_quality=float(item.get("depth_inlier_ratio", 0.0)),
                base_x=None if xyz is None else float(xyz[0]),
                base_y=None if xyz is None else float(xyz[1]),
                base_z=None if xyz is None else float(xyz[2]),
                tool_distance=(
                    None if xyz is None or current_xyz is None
                    else math.dist(tuple(map(float, xyz[:3])), current_xyz)
                ),
                selectable=bool(item.get("selectable", True)),
                unavailable_reason=str(item.get("unavailable_reason", "")),
                frame_stamp=int(item.get("frame_stamp_ns", 0)),
                result_seq=int(item.get("result_seq", 0)),
                detection_index=int(item["index"]),
            ))
        return tuple(entities)

    @staticmethod
    def _snapshot_selection(invocation: CapabilityInvocation, entities):
        by_id = {entity.entity_id: entity for entity in entities}
        requested = (*invocation.source_entity_ids,) + (
            (invocation.destination_entity_id,) if invocation.destination_entity_id else ()
        )
        missing = [entity_id for entity_id in requested if entity_id not in by_id]
        if missing:
            raise PreviewFailure("SCENE_EXPIRED", f"unknown entity IDs: {', '.join(missing)}")
        sources = [by_id[entity_id] for entity_id in invocation.source_entity_ids]
        destination = (
            [by_id[invocation.destination_entity_id]]
            if invocation.destination_entity_id else []
        )
        if any(entity.role != "pickable" for entity in sources):
            raise PreviewFailure("MODEL_RESPONSE_INVALID", "source entity role is not pickable")
        if destination and destination[0].role != "destination":
            raise PreviewFailure(
                "MODEL_RESPONSE_INVALID",
                "destination entity role is not destination",
            )
        unavailable = next(
            (entity for entity in sources + destination if not entity.selectable),
            None,
        )
        if unavailable is not None:
            raise PreviewFailure(
                unavailable.unavailable_reason or "VISION_NO_SELECTABLE_TARGET",
                f"entity {unavailable.entity_id} is not executable",
            )
        return sources, destination

    def _rebind_scene_entities(self, frozen, current, *, require_all=True):
        """Match snapshot entities to current detections without exposing indices."""
        remaining = list(current)
        rebound = []
        for old in frozen:
            candidates = [
                item for item in remaining
                if item.role == old.role and item.class_name == old.class_name
            ]
            if not candidates:
                if require_all:
                    raise PreviewFailure("TARGET_NOT_FOUND", f"entity {old.entity_id} disappeared")
                continue

            def distance(item):
                if old.base_x is not None and old.base_y is not None and item.base_x is not None:
                    return math.hypot(item.base_x - old.base_x, item.base_y - old.base_y)
                return math.hypot(item.image_u - old.image_u, item.image_v - old.image_v)

            match = min(candidates, key=distance)
            if (
                old.base_x is not None
                and match.base_x is not None
                and distance(match) > self.batch_rebind_max_distance_m
            ):
                raise PreviewFailure(
                    "TARGET_NOT_FOUND",
                    f"entity {old.entity_id} moved beyond rebind tolerance",
                )
            remaining.remove(match)
            rebound.append(match)
        return rebound

    def _compile_invocation(self, invocation, snapshot_entities, metadata):
        frozen_sources, frozen_destination = self._snapshot_selection(
            invocation, snapshot_entities
        )
        current = self._scene_entities(metadata)
        sources = self._rebind_scene_entities(frozen_sources, current)
        destinations = self._rebind_scene_entities(frozen_destination, current)
        return task_plan_for_invocation(invocation, sources, destinations)

    def _exclude_box_contents(self, metadata):
        boxes = [
            item.get("container_base_xyz") or item.get("base_xyz")
            for item in metadata
            if item.get("class_name") in self.place_classes
            and (item.get("container_base_xyz") or item.get("base_xyz"))
        ]
        radius = self.batch_box_zone_radius_m
        return [
            item for item in metadata
            if item.get("class_name") not in self.pick_classes
            or not item.get("base_xyz")
            or all(
                math.hypot(
                    float(item["base_xyz"][0]) - float(box[0]),
                    float(item["base_xyz"][1]) - float(box[1]),
                ) > radius
                for box in boxes
            )
        ]

    def _batch_trace(self, **values):
        self.get_logger().debug(
            "BATCH_TRACE " + json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        )

    def _resolve_batch_destination(self, raw_metadata, frozen_box_xyz, frame):
        boxes = [
            item for item in raw_metadata
            if item.get("class_name") in self.place_classes
        ]
        positioned = [
            item for item in boxes
            if item.get("container_base_xyz") or item.get("base_xyz")
        ]
        box_item = min(
            positioned,
            key=lambda item: math.hypot(
                float((item.get("container_base_xyz") or item["base_xyz"])[0])
                - float(frozen_box_xyz[0]),
                float((item.get("container_base_xyz") or item["base_xyz"])[1])
                - float(frozen_box_xyz[1]),
            ),
            default=None,
        )
        if box_item is not None:
            destination, reason = self.perception.resolve_candidate_detailed(
                int(box_item["index"]), frame
            )
            reason = reason or (
                "ready" if destination is not None else "VISION_DEPTH_INVALID"
            )
            return destination, reason, len(boxes)
        if boxes:
            unavailable = {
                str(item.get("unavailable_reason", "")) for item in boxes
            }
            reason = next((
                code for code in (
                    "VISION_BOX_NO_FREE_REGION", "VISION_DEPTH_INVALID",
                    "VISION_TF_UNAVAILABLE", "VISION_OBB_AXIS_INVALID",
                )
                if code in unavailable
            ), "VISION_DEPTH_INVALID")
            return None, reason, len(boxes)
        return None, "BATCH_DESTINATION_NOT_VISIBLE", 0

    def _raise_batch_rebind_timeout(
        self,
        record,
        ordinal,
        total,
        source_result,
        source_stable_frames,
        destination_reason,
        destination_stable_frames,
        box_detection_count,
    ):
        source_ready = source_stable_frames >= self.batch_rebind_stable_frames
        destination_ready = (
            destination_stable_frames >= self.batch_rebind_stable_frames
        )
        failure_stage = (
            "source_and_destination"
            if not source_ready and not destination_ready
            else "source" if not source_ready else "destination"
        )
        self._batch_trace(
            session_id=record.session_id,
            preview_id=record.preview.preview_id,
            ordinal=int(ordinal),
            total=int(total),
            completed=int(ordinal - 1),
            source_result=source_result,
            source_stable_frames=source_stable_frames,
            destination_result=destination_reason,
            destination_stable_frames=destination_stable_frames,
            box_detection_count=box_detection_count,
            failure_stage=failure_stage,
            action_result="rebind_timeout",
        )
        completed = int(ordinal - 1)
        timeout = f"{self.batch_rebind_timeout_sec:g}"
        if source_ready and destination_reason == "BATCH_DESTINATION_NOT_VISIBLE":
            detail = (
                f"已完成 {completed} 次抓放；剩余目标稳定，但盒子连续 "
                f"{timeout} 秒未被实时检测到，当前批量任务已终止。"
            )
        elif source_ready:
            detail = (
                f"已完成 {completed} 次抓放；剩余目标稳定，但盒子当前不能安全放置"
                f"（{destination_reason}），当前批量任务已终止。"
            )
        elif destination_ready:
            detail = (
                f"已完成 {completed} 次抓放；盒子可用，但剩余目标连续 "
                f"{timeout} 秒未能稳定匹配，当前批量任务已终止。"
            )
        else:
            detail = (
                f"已完成 {completed} 次抓放；剩余目标未能稳定匹配，且盒子当前不可用"
                f"（{destination_reason}），当前批量任务已终止。"
            )
        error_code = (
            destination_reason if not destination_ready
            else "BATCH_SOURCE_NOT_STABLE"
        )
        raise PreviewFailure(error_code, detail)

    def _rebind_batch_action(
        self, record, frozen_action, remaining_actions, ordinal, total
    ):
        if not self._refresh_yolo_health():
            raise PreviewFailure(
                "VISION_UNAVAILABLE", "YOLO inference is unavailable for batch rebind"
            )
        frozen = [action["source"] for action in remaining_actions]
        planned_result_seq = max(
            (int(getattr(source, "result_seq", 0)) for source in frozen),
            default=0,
        )
        deadline = time.monotonic() + max(0.0, self.batch_rebind_timeout_sec)
        last_stamp = -1
        source_stable_frames = 0
        destination_stable_frames = 0
        matched = None
        source_result = "BATCH_SOURCE_NOT_STABLE"
        destination_reason = "BATCH_DESTINATION_NOT_VISIBLE"
        box_detection_count = 0
        frozen_destination = frozen_action["destination"]
        frozen_box_xyz = frozen_destination.container_xyz or frozen_destination.xyz
        while time.monotonic() <= deadline:
            frame = self.perception.current_frame()
            stamp = -1 if frame is None else int(frame.get("stamp_ns", -1))
            result_seq = 0 if frame is None else int(frame.get("result_seq", 0))
            if (
                frame is None
                or stamp <= last_stamp
                or (planned_result_seq > 0 and result_seq <= planned_result_seq)
            ):
                time.sleep(0.05)
                continue
            last_stamp = stamp
            raw_metadata = self.perception.planning_metadata(
                frame, include_unavailable=True
            )
            metadata = self._exclude_box_contents(raw_metadata)
            current = [
                resolved_candidate(item) for item in metadata
                if item.get("class_name") in self.pick_classes
                and item.get("base_xyz")
            ]
            candidate_match = match_batch_targets(
                frozen, current, self.batch_rebind_max_distance_m
            )
            source_result = "ready" if candidate_match else "BATCH_SOURCE_NOT_STABLE"
            source_stable_frames = (
                source_stable_frames + 1 if candidate_match else 0
            )
            destination, destination_reason, box_detection_count = (
                VisualTaskPlannerMixin._resolve_batch_destination(
                    self, raw_metadata, frozen_box_xyz, frame
                )
            )
            destination_ready = destination is not None
            destination_stable_frames = (
                destination_stable_frames + 1 if destination_ready else 0
            )
            match_distances = []
            if candidate_match:
                assignments, costs = candidate_match
                match_distances = [
                    round(costs[row][assignments[row]], 4)
                    for row in range(len(frozen))
                ]
            counts = {}
            for item in raw_metadata:
                class_name = str(item.get("class_name", ""))
                counts[class_name] = counts.get(class_name, 0) + 1
            self._batch_trace(
                session_id=record.session_id,
                preview_id=record.preview.preview_id,
                ordinal=int(ordinal),
                total=int(total),
                frame_stamp_ns=stamp,
                detected_class_counts=counts,
                match_distances_xy=match_distances,
                source_result=source_result,
                source_stable_frames=source_stable_frames,
                destination_result=destination_reason,
                destination_stable_frames=destination_stable_frames,
                box_detection_count=box_detection_count,
                required_stable_frames=self.batch_rebind_stable_frames,
                action_result="rebind_wait",
            )
            matched = None
            if candidate_match and destination_ready:
                matched = (current, destination, *candidate_match)
                if (
                    source_stable_frames >= self.batch_rebind_stable_frames
                    and destination_stable_frames >= self.batch_rebind_stable_frames
                ):
                    break
            time.sleep(0.05)
        source_ready = source_stable_frames >= self.batch_rebind_stable_frames
        destination_ready = (
            destination_stable_frames >= self.batch_rebind_stable_frames
        )
        if matched is None or not source_ready or not destination_ready:
            VisualTaskPlannerMixin._raise_batch_rebind_timeout(
                self,
                record,
                ordinal,
                total,
                source_result,
                source_stable_frames,
                destination_reason,
                destination_stable_frames,
                box_detection_count,
            )
        current, destination, assignments, costs = matched
        source = current[assignments[0]]
        rebound = {
            **frozen_action,
            "source": source,
            "destination": destination,
        }
        for pose in self._pick_place_preview_poses(
            rebound["source"], rebound["destination"]
        ).values():
            self._check_pose(pose)
        self._batch_trace(
            session_id=record.session_id,
            preview_id=record.preview.preview_id,
            ordinal=int(ordinal),
            total=int(total),
            frozen_source_index=int(frozen_action["source"].index),
            current_detection_index=int(source.index),
            match_distance_xy=round(costs[0][assignments[0]], 4),
            destination_mode="rebound",
            destination_detection_index=int(destination.index),
            action_result="rebound",
        )
        return rebound

    def _workspace_ok(self, xyz):
        x, y = (float(value) for value in xyz[:2])
        return all(lower <= value <= upper for value, lower, upper in zip(
            (x, y), self.workspace_min_xy, self.workspace_max_xy
        ))

    def _pose_from_xyz_quat(self, xyz, quat):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in xyz)
        pose.pose.orientation.x, pose.pose.orientation.y = float(quat[0]), float(quat[1])
        pose.pose.orientation.z, pose.pose.orientation.w = float(quat[2]), float(quat[3])
        return pose

    def _check_pose(self, pose):
        xyz = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        if not self._workspace_ok(xyz):
            raise PreviewFailure(
                "TARGET_UNREACHABLE", f"target outside workspace whitelist: {xyz}"
            )
        quat = (pose.pose.orientation.x, pose.pose.orientation.y,
                pose.pose.orientation.z, pose.pose.orientation.w)
        if self.moveit2_arm.compute_ik(xyz, quat, wait_for_server_timeout_sec=1.0) is None:
            raise PreviewFailure(
                "TARGET_UNREACHABLE", "target pose has no collision-aware IK solution"
            )

    def _enrich_plan(self, plan: TaskPlan):
        frame = self.perception.current_frame()
        current_pose = self._current_pose()
        relative_base_known = True
        enriched = []
        for action in plan.actions:
            action_type = action["type"]
            if action_type in ("pick", "place", "pick_place"):
                with self._lock:
                    held_source = self._held_source
                validate_visual_state(
                    action_type,
                    holding=held_source is not None,
                    recovery=False,
                )
                if action_type in ("pick", "pick_place"):
                    source, reason = self.perception.resolve_candidate_detailed(
                        action["source_index"], frame
                    )
                    if source is None:
                        raise PreviewFailure(
                            reason, f"selected pick target is unavailable: {reason}"
                        )
                else:
                    source = held_source
                destination = None
                if action_type in ("place", "pick_place"):
                    destination, reason = self.perception.resolve_candidate_detailed(
                        action["destination_index"], frame
                    )
                    if destination is None:
                        raise PreviewFailure(
                            reason, f"selected box is unavailable: {reason}"
                        )
                enriched_action = {**action, "source": source}
                if destination is not None:
                    enriched_action["destination"] = destination
                enriched.append(enriched_action)
                if action_type == "pick":
                    poses = self._pick_preview_poses(source)
                elif action_type == "place":
                    poses = self._place_preview_poses(source, destination)
                else:
                    poses = self._pick_place_preview_poses(source, destination)
                for pose in poses.values():
                    self._check_pose(pose)
                relative_base_known = False
            elif action_type == "move_relative":
                if not relative_base_known:
                    raise PreviewFailure(
                        "PLAN_INVALID",
                        "move_relative after a visual action is unsafe because its "
                        "execution-time reference pose is not known",
                    )
                q = current_pose.pose.orientation
                current_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
                delta = Rotation.from_euler(
                    "xyz",
                    [action["droll_deg"], action["dpitch_deg"], action["dyaw_deg"]],
                    degrees=True,
                )
                quat = (delta * current_rotation).as_quat()
                xyz = (
                    current_pose.pose.position.x + action["dx"],
                    current_pose.pose.position.y + action["dy"],
                    current_pose.pose.position.z + action["dz"],
                )
                pose = self._pose_from_xyz_quat(xyz, quat)
                self._check_pose(pose)
                enriched.append({**action, "target_pose": pose})
                current_pose = pose
                relative_base_known = True
            elif action_type == "set_gripper":
                enriched.append(dict(action))
            else:
                raise PreviewFailure(
                    "MODEL_RESPONSE_INVALID", f"unsupported action: {action_type}"
                )
        return enriched

    def _grasp_quat(self, source):
        roll, pitch, yaw_offset = self.grasp_profiles[source.class_name]
        return Rotation.from_euler(
            "xyz", [roll, pitch, math.degrees(source.yaw) + yaw_offset], degrees=True
        ).as_quat()

    def _pick_heights(self, source):
        return (
            source.xyz[2] + self.grasp_offset,
            source.xyz[2] + self.grasp_above,
            source.xyz[2] + self.place_offset,
        )

    def _pick_preview_poses(self, source):
        grasp, approach, carry = self._pick_heights(source)
        quat = self._grasp_quat(source)
        return {
            "approach_pick": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], approach), quat
            ),
            "grasp": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], grasp), quat
            ),
            "carry": self._pose_from_xyz_quat(
                (source.xyz[0], source.xyz[1], carry), quat
            ),
        }

    def _place_preview_poses(self, source, destination):
        if source is None or getattr(source, "class_name", "") not in self.grasp_profiles:
            orientation = self.pregrasp_pose.pose.orientation
            quat = (orientation.x, orientation.y, orientation.z, orientation.w)
        else:
            quat = self._grasp_quat(source)
        return {
            "approach_box": self._pose_from_xyz_quat(
                (
                    destination.xyz[0],
                    destination.xyz[1],
                    destination.xyz[2] + self.place_offset,
                ),
                quat,
            ),
            "release": self._pose_from_xyz_quat(
                (
                    destination.xyz[0],
                    destination.xyz[1],
                    destination.xyz[2] + self.descend_to_box,
                ),
                quat,
            ),
        }

    def _pick_place_preview_poses(self, source, destination):
        return {
            **self._pick_preview_poses(source),
            **self._place_preview_poses(source, destination),
        }

    def _build_pregrasp_pose(self):
        cfg = self.pregrasp_pose_cfg
        quat = Rotation.from_euler(
            "xyz", [cfg["roll"], cfg["pitch"], cfg["yaw"]], degrees=True
        ).as_quat()
        return self._pose_from_xyz_quat((cfg["x"], cfg["y"], cfg["z"]), quat)

    def _build_preview(self, plan, session_id, preview_epoch):
        if any(action["type"] == "place" for action in plan.actions):
            with self._lock:
                self._state = LlmControlTaskState.SEARCHING_BOX.value
        enriched = self._enrich_plan(plan)
        visual_actions = [
            action for action in plan.actions
            if action["type"] in ("pick", "place", "pick_place")
        ]
        preview_id = uuid.uuid4().hex
        preview = TaskPreview(
            preview_id, plan, time.monotonic(), self.preview_max_age_sec
        )
        record = PreviewRecord(
            preview, session_id, enriched, preview_epoch
        )
        with self._lock:
            motion_block_reason = self._motion_block_reason_locked()
            if (
                not safety_execution_valid(self._safety, preview_epoch)
                or motion_block_reason
                or self._state in ("STOPPED", "RESETTING", "EXECUTING")
            ):
                detail = f": {motion_block_reason}" if motion_block_reason else ""
                raise PreviewFailure(
                    "SAFETY_BLOCKED",
                    f"motion safety state changed while preview was generated{detail}",
                )
            self._previews[preview_id] = record
            self._clear_task_context_locked(session_id)
            self._state = LlmControlTaskState.PREVIEW_READY.value
        if len(visual_actions) > 1:
            self._batch_trace(
                session_id=session_id,
                preview_id=preview_id,
                frozen_target_count=len(visual_actions),
                action_result="preview_ready",
            )
        return preview_id
