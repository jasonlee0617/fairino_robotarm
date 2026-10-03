import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

from visual_perception_utils.model_utils import (  # noqa: E402
    assign_obb_confidence,
    FOUR_CLASS_OBB_NAMES,
    POSITION_3D_TOPICS,
    AXIS_3D_TOPICS,
    require_four_class_obb_model,
)
from visual_perception_utils.visualization import draw_detection_diagnostics  # noqa: E402
from visual_perception_utils.llm_rgbd import (  # noqa: E402
    ResolvedCandidate,
    RgbdPerception,
)


ROOT = Path(__file__).resolve().parents[1]


def test_assigns_obb_box_confidence_to_inference_result():
    inference_result = SimpleNamespace(confidence=0.0)
    box = SimpleNamespace(conf=np.array([0.875], dtype=np.float32))

    assign_obb_confidence(inference_result, box)

    assert isinstance(inference_result.confidence, float)
    assert inference_result.confidence == pytest.approx(0.875)


def test_rgbd_frame_sequence_changes_only_for_a_new_synchronized_pair():
    def message(stamp_ns):
        stamp = SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        )
        return SimpleNamespace(header=SimpleNamespace(stamp=stamp))

    perception = RgbdPerception.__new__(RgbdPerception)
    perception.rgb_depth_tolerance_sec = 0.05
    perception._yolo_frames = deque([message(1_000_000_000)], maxlen=20)
    perception._depth_frames = deque(
        [(message(1_020_000_000).header, np.zeros((2, 2), dtype=np.float32))],
        maxlen=20,
    )
    perception._active_frame = None
    perception._result_seq = 0

    perception._activate_frame_locked()
    first = perception._active_frame
    perception._activate_frame_locked()

    assert first["result_seq"] == 1
    assert perception._active_frame is first

    perception._yolo_frames.append(message(2_000_000_000))
    perception._depth_frames.append(
        (message(2_010_000_000).header, np.zeros((2, 2), dtype=np.float32))
    )
    perception._activate_frame_locked()
    assert perception._active_frame["result_seq"] == 2


def test_resolved_candidate_public_metadata_remains_stable_after_split():
    candidate = ResolvedCandidate(
        index=2, class_name="cube", confidence=0.9,
        center_uv=(10.0, 20.0), xyz=(0.1, 0.2, 0.3), yaw=0.4,
        frame_stamp_ns=42, depth_inlier_ratio=0.8, result_seq=7,
    )
    value = candidate.public()
    assert value["base_xyz"] == [0.1, 0.2, 0.3]
    assert value["result_seq"] == 7 and value["selectable"] is True


def test_accepts_only_the_four_class_obb_contract():
    assert require_four_class_obb_model(
        ["box", "elongated_object", "cube", "stone"]
    ) == FOUR_CLASS_OBB_NAMES


def test_rejects_legacy_three_class_gazebo_contract():
    with pytest.raises(ValueError, match="legacy yolo-obb-gazebo"):
        require_four_class_obb_model({0: "pen", 1: "box", 2: "cube"})


def test_four_class_results_route_to_their_semantic_topics():
    assert POSITION_3D_TOPICS == {
        "box": "/box_position_3d",
        "elongated_object": "/elongated_object_position_3d",
        "cube": "/cube_position_3d",
        "stone": "/stone_position_3d",
    }
    assert AXIS_3D_TOPICS["elongated_object"] == "/elongated_object_axis_3d"
    assert AXIS_3D_TOPICS["cube"] == "/cube_axis_3d"
    assert AXIS_3D_TOPICS["stone"] == "/stone_axis_3d"


def test_llm_perception_uses_the_new_result_topics_only():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()
    cmake = (ROOT / "CMakeLists.txt").read_text()

    assert '"/yolo/detected_result"' in source
    assert '"/yolo/detected_result/depth"' in source
    assert '"/camera/detected_result"' in source
    assert "/" + "Yolov8" + "_Inference" not in source
    assert "results[0].plot()" not in source
    assert "draw_detection_center" in source
    assert "draw_obb_major_axis" in source
    assert "ReliabilityPolicy.RELIABLE" in source
    helper = ROOT / "visual_perception_utils" / "llm_rgbd.py"
    assert helper.exists()
    assert "class RgbdPerception" in helper.read_text()
    assert "class RgbdPerception" not in source
    assert (
        "ament_python_install_package(visual_perception_nodes PACKAGE_DIR visual_perception/nodes)"
        in cmake
    )


def test_llm_perception_entrypoint_is_executable():
    assert os.access(ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py", os.X_OK)


def test_llm_perception_launch_treats_visual_node_as_critical():
    source = (ROOT / "launch" / "llm_visual_perception.launch.py").read_text()

    assert "require_cuda_for_visual_perception" in source
    assert "OpaqueFunction(function=_validate_cuda)" in source
    assert "on_exit=EmitEvent" in source
    assert "Shutdown" in source


def test_llm_perception_is_continuous_and_has_one_health_service():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()

    assert "device=self.inference_device" in source
    assert "require_cuda_for_visual_perception" in source
    assert "torch.cuda.is_available()" in source
    assert "inference_ms=" in source
    assert "use_continuous_yolo" not in source
    assert "/llm_visual_perception/set_inference_enabled" not in source
    assert "/llm_visual_perception/release_gpu" not in source
    assert "/llm_visual_perception/status" in source
    assert "self._visual_last_error" in source
    assert "LLM YOLO inference disabled after CUDA OOM" in source
    assert "def _is_cuda_oom" in source
    assert "torch.cuda.empty_cache()" in source
    assert "if self.model is None or not self._visual_available:" in source
    assert "MultiThreadedExecutor(num_threads=2)" in source
    assert "self._shutting_down or self._inference_active.is_set()" in source
    assert "self._inference_active.clear()" in source
    assert "now - self._last_sync_warning_at >= 30.0" in source
    assert "self._latest_pair = (rgb_msg, depth_msg)" in source
    assert "callback_group=self._inference_callback_group" in source
    assert "recreating camera subscriptions" not in source
    assert "VISION_SYNC " in source


def test_llm_perception_shutdown_stops_callbacks_before_model_unload():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()

    assert "def _begin_shutdown(self):" in source
    assert "self.inference_timer.cancel()" in source
    assert "self.sync_watchdog.cancel()" in source
    assert source.index("node._begin_shutdown()") < source.index(
        "executor.shutdown(timeout_sec=2.0)"
    )
    assert source.index("executor.shutdown(timeout_sec=2.0)") < source.index(
        "node._unload_model()"
    )


def test_llm_perception_launch_wires_rgbd_sync_parameters():
    launch = (ROOT / "launch" / "llm_visual_perception.launch.py").read_text()

    assert 'DeclareLaunchArgument("sync_slop", default_value="0.05")' in launch
    assert '"sync_slop": LaunchConfiguration("sync_slop")' in launch
    assert '"sync_watchdog_sec": LaunchConfiguration("sync_watchdog_sec")' in launch
    assert '"expected_camera_rate_hz": LaunchConfiguration(' in launch


def test_obb_detector_uses_the_same_inference_gate_contract():
    source = (ROOT / "visual_perception" / "nodes" / "yolo_detector_obb.py").read_text()

    assert "use_continuous_yolo" in source
    assert "/yolo_detector_obb/set_inference_enabled" in source
    assert "if not self._inference_enabled:" in source


def test_detection_diagnostics_draws_text():
    image = np.zeros((80, 160, 3), dtype=np.uint8)

    draw_detection_diagnostics(image, (10, 10), ["cube conf=0.90"], (0, 255, 0))

    assert image.any()


def test_llm_perception_source_includes_complete_3d_diagnostics():
    source = (ROOT / "visual_perception" / "nodes" / "llm_visual_perception.py").read_text()

    assert "draw_detection_diagnostics" in source
    assert 'f"base: {center_base.point.x:.3f},' in source
    assert "depthQ:" in source
    assert "3D unavailable:" in source
