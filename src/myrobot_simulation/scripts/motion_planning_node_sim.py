#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 交互式仿真规划与 IK 对比节点
#
# 提供交互式终端控制，支持：
#   - 输入目标位姿 (x y z 或 x y z rx ry rz) 进行规划与运动
#   - 切换 IK 求解器 (fairino / kdl)
#   - 切换规划算法 (mire_biait*, birrt*, rrt, rrt*, aapf_birrt* 等)
#   - 返回配置起点、重置场景 (recover)
#   - 末端轨迹实时可视化 (RViz Marker)
#
# 依赖：
#   - pymoveit2 (MoveIt2 Python 接口)
#   - pathplanning_scene_tools (场景加载管理)
#   - scipy (RPY→四元数转换)
# ---------------------------------------------------------------------------

import csv
import hashlib
import json
import math
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from typing import List, Optional, Tuple

import yaml

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState, RobotTrajectory
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from ament_index_python.packages import get_package_share_directory
from pymoveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
from pathplanning_scene_tools import SceneEnvironmentManager
from myrobot_common.planning.motion_executor import PlannerSwitch
from planning_benchmark import (
    GoalSetSpec,
    ALGORITHM_DIAGNOSTIC_FIELDS,
    ANYTIME_FIELDS,
    ROOT_DIAGNOSTIC_FIELDS,
    TRAJECTORY_PATH_FIELDS,
    adaptive_challenge_metrics,
    benchmark_slug,
    build_goal_sampling_report,
    canonical_sha256,
    distance_to_obstacle_surface,
    sampling_identity_payload,
    goal_bounds,
    goal_is_separated,
    iter_random_candidates,
    load_goal_collection,
    obstacle_signature,
    prepare_benchmark_run,
    finalize_run_manifest,
    write_benchmark_summary,
    initialize_standard_csvs,
    validate_complete_run,
    write_goal_collection,
    write_csv_atomic,
    write_results,
)
from planning_motion import execute_joint_trajectory, joint_trajectory_path_length
from planning_trace import append_trace_point

import tf2_ros
from tf2_ros import TransformException


BENCHMARK_PLANNER_IDS = {
    "rrt*", "informed_rrt*", "birrt*", "aapf_birrt*",
    "mire_biait*", "prm",
}


class MotionPlanningNodeSim(Node):
    """ROS2 节点：交互式路径规划与 Fairino/KDL IK 对比。"""

    def __init__(self):
        super().__init__("motion_planning_node_sim")

        # 使用可重入回调组，允许多个回调并发执行
        self.callback_group = ReentrantCallbackGroup()

        # ═══════════════════════════════════════════════════════
        #  声明 ROS 参数（机器人、规划、场景）
        # ═══════════════════════════════════════════════════════

        # 机器人基础参数
        self.declare_parameter("planning_client", "fairino")
        self.declare_parameter("move_group_namespace", "")
        self.declare_parameter("group_name", "robot_arm")
        self.declare_parameter("base_frame_name", "base_link")
        self.declare_parameter("ee_frame_name", "tool0")
        self.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6")
        self.declare_parameter("start_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")
        self.declare_parameter("start_id", "home")
        self.declare_parameter("start_settle_timeout_s", 6.0)
        self.declare_parameter("ik_timeout", 3.0)

        # 规划参数
        self.declare_parameter("default_pipeline_id", "fairino")
        self.declare_parameter("default_planner_id", "birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")  # 默认末端姿态（RPY 度）
        self.declare_parameter("go_start_before_demo", False)
        self.declare_parameter("allowed_planning_time", 30.0)

        # 场景与障碍物参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", True)
        self.declare_parameter("scene_config_file", "")
        self.declare_parameter("scene_name", "single_obstacle")
        self.declare_parameter("scene_assets_dir", "")
        self.declare_parameter("spawn_sim_scene_models", False)
        self.declare_parameter("sim_world", "empty")
        self.declare_parameter("publish_planning_scene", True)
        self.declare_parameter("publish_obstacle_markers", True)
        self.declare_parameter("obstacle_marker_topic", "/demo_pathplanning/obstacle_markers")
        self.declare_parameter("planning_scene_obstacle_padding_m", 0.03)

        # benchmark 配置保持 YAML 专属；唯一的运行时归档入口由 launch 注入。
        self.declare_parameter("run_mode", "interactive")
        self.declare_parameter("goal_collection_dir", "")
        self.declare_parameter("benchmark_output_dir", "")
        self.declare_parameter("benchmark_goal_root_mode", "single_root")
        self.declare_parameter("benchmark_effective_config_json", "{}")
        self.declare_parameter("benchmark_repetitions", 20)
        self.declare_parameter("benchmark_startup_joint_state_timeout_s", 90.0)
        self.declare_parameter("benchmark_goal_mode", "adaptive_obstacle_challenge_region")
        self.declare_parameter("ik_mode", "continuous")
        self.declare_parameter("benchmark_goal_seed", 17)
        self.declare_parameter("planner_random_seed", 7)
        self.declare_parameter("benchmark_variant", "full")
        self.declare_parameter("benchmark_goal_clearance_min_m", 0.06)
        self.declare_parameter("benchmark_goal_clearance_max_m", 0.14)
        self.declare_parameter("benchmark_goal_corridor_clearance_max_m", 0.10)
        self.declare_parameter("benchmark_goal_min_separation_m", 0.04)
        self.declare_parameter("benchmark_goal_state_validity_timeout_s", 2.0)

        # 等待参数服务器就绪
        time.sleep(2.0)

        # 解析参数并初始化
        self.setup_params()
        self.setup_moveit()
        self.setup_ik_comparison()
        self.setup_ee_trace()

        # 发布任务状态（自定义消息）
        self.state_publisher = self.create_publisher(String, "/task_state", 10)
        self.display_trajectory_pub = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )

        self.get_logger().info("交互式规划与 IK 对比节点启动完成")

    # ═══════════════════════════════════════════════════════
    #  通用解析工具（静态方法）
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _parse_str_list(value) -> List[str]:
        """将逗号/分号分隔的字符串转为列表。"""
        if isinstance(value, (list, tuple)):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).replace(";", ",").split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        """将逗号/分号/空格分隔的数字字符串转为浮点数列表。"""
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).replace(";", ",").replace(" ", ",")
        return [float(v) for v in text.split(",") if v.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        """解析布尔值（支持 1/true/yes 等字符串）。"""
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def _resolve_benchmark_output_dir(value) -> str:
        """Expand a user-facing archive path once before any case I/O."""
        path = str(value).strip()
        return os.path.abspath(os.path.expandvars(os.path.expanduser(path))) if path else ""

    @staticmethod
    def _normalize_benchmark_goal_mode(value: str) -> str:
        key = str(value).strip().lower()
        if key == "adaptive":
            return "adaptive_obstacle_challenge_region"
        if key != "adaptive_obstacle_challenge_region":
            raise ValueError("benchmark_goal_mode 仅支持 adaptive_obstacle_challenge_region")
        return key

    @staticmethod
    def _pose_quat_from_rpy(rpy_deg):
        """将 RPY 欧拉角（度）转换为四元数 (x, y, z, w)。"""
        quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        return tuple(float(v) for v in quat)

    @classmethod
    def _parse_pose_values(cls, values, fallback_rpy_deg):
        """解析位姿值：3 个数字为 (x,y,z) 使用默认姿态，6 个数字为 (x,y,z,r,p,y)。"""
        if len(values) == 3:
            xyz = tuple(float(v) for v in values)
            rpy = tuple(float(v) for v in fallback_rpy_deg)
            return xyz, rpy
        if len(values) == 6:
            xyz = tuple(float(v) for v in values[:3])
            rpy = tuple(float(v) for v in values[3:])
            return xyz, rpy
        raise ValueError("pose input must contain 3 or 6 values")

    # ═══════════════════════════════════════════════════════
    #  参数设置与校验
    # ═══════════════════════════════════════════════════════

    def setup_params(self):
        """读取并存储所有 ROS 参数，初始化场景管理器。"""
        self.group_name = str(self.get_parameter("group_name").value)
        self.base_frame_name = str(self.get_parameter("base_frame_name").value)
        self.ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        self.joint_names = self._parse_str_list(self.get_parameter("joint_names").value)
        self.start_joints = tuple(
            self._parse_float_list(self.get_parameter("start_joints").value)
        )
        if len(self.start_joints) != len(self.joint_names):
            raise ValueError("start_joints must contain one value for each joint_names entry")
        self.start_id = str(self.get_parameter("start_id").value).strip()
        if not self.start_id or not self.start_id.isascii() or not self.start_id.isalnum():
            raise ValueError("start_id must contain only ASCII letters and digits")
        self.start_xyz = None
        self.start_joint_state = None

        self.default_pipeline_id = str(self.get_parameter("default_pipeline_id").value)
        self.default_planner_id = PlannerSwitch.normalize_planner(
            self.default_pipeline_id,
            str(self.get_parameter("default_planner_id").value),
        )
        self.default_planning_client = PlannerSwitch.normalize_ik(
            str(self.get_parameter("planning_client").value)
        )
        self.allowed_planning_time = max(
            1e-3, float(self.get_parameter("allowed_planning_time").value)
        )
        self.go_start_before_demo = self._as_bool(self.get_parameter("go_start_before_demo").value)
        self.start_settle_timeout_s = max(
            0.5, float(self.get_parameter("start_settle_timeout_s").value)
        )

        self.publish_planning_scene = self._as_bool(
            self.get_parameter("publish_planning_scene").value)
        self.publish_obstacle_markers = self._as_bool(
            self.get_parameter("publish_obstacle_markers").value)
        self.spawn_sim_scene_models = self._as_bool(
            self.get_parameter("spawn_sim_scene_models").value)
        self.sim_world = str(self.get_parameter("sim_world").value)
        self.obstacle_marker_topic = str(self.get_parameter("obstacle_marker_topic").value)

        # 场景资源目录
        gz_share = get_package_share_directory("myrobot_simulation")
        default_assets_dir = os.path.join(gz_share, "config", "scenes")
        self.scene_assets_dir = str(self.get_parameter("scene_assets_dir").value).strip()
        if not self.scene_assets_dir:
            self.scene_assets_dir = default_assets_dir

        self.scene_config_file = str(self.get_parameter("scene_config_file").value).strip()
        if not self.scene_config_file:
            self.scene_config_file = os.path.join(self.scene_assets_dir, "pathplanning_scenes_params.yaml")
        self.scene_name = str(self.get_parameter("scene_name").value).strip() or "single_obstacle"
        self.planning_scene_obstacle_padding_m = max(
            0.0, float(self.get_parameter("planning_scene_obstacle_padding_m").value)
        )

        self.run_mode = str(self.get_parameter("run_mode").value).strip().lower()
        if self.run_mode not in (
            "interactive", "goal_collection", "benchmark_execution", "benchmark_algorithm"
        ):
            raise ValueError(
                "run_mode must be interactive, goal_collection, benchmark_execution, or benchmark_algorithm"
            )
        self.goal_collection_dir = self._resolve_benchmark_output_dir(
            self.get_parameter("goal_collection_dir").value
        )
        self.benchmark_output_dir = self._resolve_benchmark_output_dir(
            self.get_parameter("benchmark_output_dir").value
        )
        self.benchmark_goal_root_mode = str(
            self.get_parameter("benchmark_goal_root_mode").value
        ).strip().lower()
        if self.benchmark_goal_root_mode not in ("single_root", "multi_root"):
            raise ValueError("benchmark_goal_root_mode must be single_root or multi_root")
        try:
            self.benchmark_effective_config = json.loads(str(
                self.get_parameter("benchmark_effective_config_json").value))
        except json.JSONDecodeError as exc:
            raise ValueError("benchmark_effective_config_json must be valid JSON") from exc
        if not isinstance(self.benchmark_effective_config, dict):
            raise ValueError("benchmark_effective_config_json must encode an object")
        self.is_benchmark_mode = self.run_mode in ("benchmark_execution", "benchmark_algorithm")
        self.effective_allowed_planning_time = self.allowed_planning_time
        if self.is_benchmark_mode:
            comparison = self.benchmark_effective_config.get("comparison", {})
            self.effective_allowed_planning_time = max(
                1e-3, float(comparison.get("planning_deadline_s", 15.0))
            )
        if self.run_mode == "goal_collection" and not self.goal_collection_dir:
            raise ValueError("goal_collection_dir is required for goal_collection mode")
        if self.is_benchmark_mode and not self.benchmark_output_dir:
            raise ValueError("benchmark_output_dir is required for benchmark run_mode")
        if self.is_benchmark_mode:
            if (PlannerSwitch.normalize_pipeline(self.default_pipeline_id) != "fairino" or
                    self.default_planner_id not in BENCHMARK_PLANNER_IDS):
                raise ValueError(
                    "benchmark 仅支持 fairino 的 rrt*、informed_rrt*、birrt*、aapf_birrt*、mire_biait*、prm"
                )
            self.get_logger().info(
                f"Benchmark archive root: {self.benchmark_output_dir}"
            )
        self.benchmark_repetitions = max(1, int(self.get_parameter("benchmark_repetitions").value))
        self.benchmark_startup_joint_state_timeout_s = max(
            1.0, float(self.get_parameter("benchmark_startup_joint_state_timeout_s").value)
        )
        self.benchmark_goal_mode = self._normalize_benchmark_goal_mode(
            self.get_parameter("benchmark_goal_mode").value
        )
        self.ik_mode = str(self.get_parameter("ik_mode").value).strip().lower()
        if self.run_mode in ("goal_collection", "benchmark_execution", "benchmark_algorithm") and self.ik_mode != "continuous":
            raise ValueError("目标集采集与 benchmark 的 ik_mode 仅支持 continuous")
        self.benchmark_goal_seed = int(self.get_parameter("benchmark_goal_seed").value)
        self.planner_random_seed = int(self.get_parameter("planner_random_seed").value)
        self.benchmark_variant = str(self.get_parameter("benchmark_variant").value).strip().lower() or "full"
        if self.benchmark_variant not in (
            "full", "cost_only_queue", "eager_edge_validation", "cost_only_eager"):
            raise ValueError(
                "benchmark_variant must be full, cost_only_queue, eager_edge_validation, or cost_only_eager")
        if self.default_planner_id != "mire_biait*" and self.benchmark_variant != "full":
            raise ValueError("benchmark_variant is only valid for mire_biait*")
        self.benchmark_goal_clearance_min_m = max(
            0.0, float(self.get_parameter("benchmark_goal_clearance_min_m").value)
        )
        self.benchmark_goal_clearance_max_m = max(
            self.benchmark_goal_clearance_min_m,
            float(self.get_parameter("benchmark_goal_clearance_max_m").value),
        )
        self.benchmark_goal_corridor_clearance_max_m = max(
            0.0, float(self.get_parameter("benchmark_goal_corridor_clearance_max_m").value)
        )
        self.benchmark_goal_min_separation_m = max(
            0.0, float(self.get_parameter("benchmark_goal_min_separation_m").value)
        )
        self.benchmark_goal_state_validity_timeout_s = max(
            0.1, float(self.get_parameter("benchmark_goal_state_validity_timeout_s").value)
        )
        self.benchmark_executes_trajectory = self.run_mode == "benchmark_execution"

        # 基本校验
        # 运动间默认延迟（秒）
        self.action_delay = 1.0

        self.scene_manager = None
        self.active_obstacles = []
        self._benchmark_goal_sampling_report = None

    def setup_scene(self):
        """仅在路径规划模式加载并发布场景。"""
        if self.scene_manager is not None:
            return
        self.scene_manager = SceneEnvironmentManager(
            node=self,
            base_frame_name=self.base_frame_name,
            scene_name=self.scene_name,
            scene_config_file=self.scene_config_file,
            sim_world=self.sim_world,
            obstacle_marker_topic=self.obstacle_marker_topic,
            publish_planning_scene=self.publish_planning_scene,
            publish_obstacle_markers=self.publish_obstacle_markers,
            spawn_sim_scene_models=self.spawn_sim_scene_models,
            planning_scene_obstacle_padding_m=self.planning_scene_obstacle_padding_m,
        )
        # 加载当前场景的障碍物列表
        self.active_obstacles = self.scene_manager.load_scene()
        self.scene_benchmark = getattr(self.scene_manager, "benchmark", {}) or {}

    # ═══════════════════════════════════════════════════════
    #  末端轨迹可视化
    # ═══════════════════════════════════════════════════════

    def setup_ee_trace(self):
        """初始化末端轨迹 Marker 发布器及 TF 监听。"""
        # 声明可视化专用参数
        self.declare_parameter("trace_base_frame", self.base_frame_name)
        self.declare_parameter("trace_ee_frame", self.ee_frame_name)
        self.declare_parameter("trace_marker_topic", "/demo_pathplanning/ee_trace_marker")
        self.declare_parameter("trace_marker_ns", "demo_ee_trace")
        self.declare_parameter("trace_line_width", 0.006)
        self.declare_parameter("trace_tip_size", 0.012)
        self.declare_parameter("trace_max_points", 3000)
        self.declare_parameter("trace_sample_period", 0.05)
        self.declare_parameter("trace_min_distance", 0.0015)

        # 读取参数
        self.trace_base_frame = str(self.get_parameter("trace_base_frame").value)
        self.trace_ee_frame = str(self.get_parameter("trace_ee_frame").value)
        self.trace_marker_topic = str(self.get_parameter("trace_marker_topic").value)
        self.trace_marker_ns = str(self.get_parameter("trace_marker_ns").value)
        self.trace_line_width = float(self.get_parameter("trace_line_width").value)
        self.trace_tip_size = float(self.get_parameter("trace_tip_size").value)
        self.trace_max_points = int(self.get_parameter("trace_max_points").value)
        self.trace_sample_period = float(self.get_parameter("trace_sample_period").value)
        self.trace_min_distance = float(self.get_parameter("trace_min_distance").value)

        # TF 缓存与监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ee_marker_pub = self.create_publisher(Marker, self.trace_marker_topic, 10)

        # 线段 Marker（末端轨迹连线）
        self.ee_trace_line = Marker()
        self.ee_trace_line.header.frame_id = self.trace_base_frame
        self.ee_trace_line.ns = self.trace_marker_ns
        self.ee_trace_line.id = 0
        self.ee_trace_line.type = Marker.LINE_STRIP
        self.ee_trace_line.action = Marker.ADD
        self.ee_trace_line.pose.orientation.w = 1.0
        self.ee_trace_line.scale.x = self.trace_line_width
        self.ee_trace_line.color.r = 0.1
        self.ee_trace_line.color.g = 0.9
        self.ee_trace_line.color.b = 0.2
        self.ee_trace_line.color.a = 1.0

        # 当前末端点 Marker（球体）
        self.ee_trace_tip = Marker()
        self.ee_trace_tip.header.frame_id = self.trace_base_frame
        self.ee_trace_tip.ns = self.trace_marker_ns
        self.ee_trace_tip.id = 1
        self.ee_trace_tip.type = Marker.SPHERE
        self.ee_trace_tip.action = Marker.ADD
        self.ee_trace_tip.pose.orientation.w = 1.0
        self.ee_trace_tip.scale.x = self.trace_tip_size
        self.ee_trace_tip.scale.y = self.trace_tip_size
        self.ee_trace_tip.scale.z = self.trace_tip_size
        self.ee_trace_tip.color.r = 1.0
        self.ee_trace_tip.color.g = 0.2
        self.ee_trace_tip.color.b = 0.2
        self.ee_trace_tip.color.a = 1.0

        self.last_trace_xyz = None
        # 定时更新轨迹
        self.create_timer(self.trace_sample_period, self.publish_ee_trace, callback_group=self.callback_group)

        self.get_logger().info(
            f"末端轨迹可视化已启用: marker={self.trace_marker_topic}, "
            f"frame={self.trace_base_frame}->{self.trace_ee_frame}"
        )

    def publish_ee_trace(self):
        """定时采样末端位姿，更新并发布轨迹 Marker。"""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.trace_base_frame,
                self.trace_ee_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            return

        x = float(tf_msg.transform.translation.x)
        y = float(tf_msg.transform.translation.y)
        z = float(tf_msg.transform.translation.z)
        xyz = np.array([x, y, z], dtype=float)

        # 距离过小时仅更新尖端，避免轨迹点堆积
        if self.last_trace_xyz is not None:
            if np.linalg.norm(xyz - self.last_trace_xyz) < self.trace_min_distance:
                self.ee_trace_tip.header.stamp = tf_msg.header.stamp
                self.ee_trace_tip.pose.position.x = x
                self.ee_trace_tip.pose.position.y = y
                self.ee_trace_tip.pose.position.z = z
                self.ee_marker_pub.publish(self.ee_trace_tip)
                return

        self.last_trace_xyz = xyz

        self.ee_trace_line.header.stamp = tf_msg.header.stamp
        append_trace_point(self.ee_trace_line, xyz, self.trace_max_points)

        self.ee_trace_tip.header.stamp = tf_msg.header.stamp
        self.ee_trace_tip.pose.position.x = x
        self.ee_trace_tip.pose.position.y = y
        self.ee_trace_tip.pose.position.z = z

        self.ee_marker_pub.publish(self.ee_trace_line)
        self.ee_marker_pub.publish(self.ee_trace_tip)

    def clear_ee_trace(self):
        """清除末端轨迹 Marker。"""
        self.last_trace_xyz = None
        self.ee_trace_line.points = []
        self.ee_trace_line.header.stamp = self.get_clock().now().to_msg()
        self.ee_marker_pub.publish(self.ee_trace_line)

        self.ee_trace_tip.action = Marker.DELETE
        self.ee_marker_pub.publish(self.ee_trace_tip)
        self.ee_trace_tip.action = Marker.ADD

    # ═══════════════════════════════════════════════════════
    #  MoveIt2 初始化
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _normalize_move_group_namespace(namespace: str) -> str:
        """规范化命名空间：以 '/' 开头且末尾无 '/'。"""
        ns = (namespace or "").strip()
        if not ns:
            return ""
        if not ns.startswith("/"):
            ns = f"/{ns}"
        return ns.rstrip("/")

    @staticmethod
    def _resolve_move_group_endpoint(namespace: str, endpoint: str) -> str:
        """拼接完整的服务/动作端点名称。"""
        if not namespace:
            return f"/{endpoint}"
        return f"{namespace}/{endpoint}"

    def _resolve_planning_client(self):
        """根据 planning_client 参数确定使用的 MoveIt 命名空间。"""
        planning_client = PlannerSwitch.normalize_ik(
            self.get_parameter("planning_client").get_parameter_value().string_value
        )
        namespace_override = self._normalize_move_group_namespace(
            self.get_parameter("move_group_namespace").get_parameter_value().string_value
        )

        client_to_namespace = {
            "fairino": "/move_group_fairino",
            "kdl": "/move_group_kdl",
        }

        if planning_client not in client_to_namespace:
            self.get_logger().error(
                f"非法参数 planning_client='{planning_client}'，仅支持 fairino 或 kdl。"
            )
            raise ValueError("invalid planning_client")

        if namespace_override:
            client_to_namespace[planning_client] = namespace_override
        return planning_client, client_to_namespace, namespace_override

    def _make_moveit2_arm(self, move_group_namespace: str):
        return MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_frame_name,
            end_effector_name=self.ee_frame_name,
            group_name=self.group_name,
            callback_group=self.callback_group,
            use_move_group_action=True,
            move_group_namespace=move_group_namespace,
        )

    def _configure_moveit2_arm(self, arm):
        arm.pipeline_id = self.default_pipeline_id
        arm.planner_id = self.default_planner_id
        arm.max_velocity = 0.5
        arm.max_acceleration = 0.5
        arm.allowed_planning_time = self.effective_allowed_planning_time
        arm.goal_position_tolerance = 0.001
        arm.goal_orientation_tolerance = 0.01
        arm.max_step = 0.01
        arm.jump_threshold = 0.0

    def setup_moveit(self):
        """初始化 MoveIt2 客户端并设置运动参数。"""
        try:
            planning_client, move_group_namespaces, namespace_override = self._resolve_planning_client()
            self.move_group_namespaces = move_group_namespaces
            self.moveit2_arms = {
                client: self._make_moveit2_arm(namespace)
                for client, namespace in move_group_namespaces.items()
            }
            for arm in self.moveit2_arms.values():
                self._configure_moveit2_arm(arm)
            if not self.set_ik(planning_client):
                raise ValueError("invalid planning_client")
            self.get_logger().info("MoveIt接口初始化成功")
            self.get_logger().info(f"  规划管线: {self.moveit2_arm.pipeline_id}")
            self.get_logger().info(f"  规划算法: {self.moveit2_arm.planner_id}")
            self.get_logger().info(
                f"  规划客户端: {planning_client}, 命名空间: {self.move_group_namespace}, "
                f"override={'yes' if namespace_override else 'no'}"
            )
            # 输出关键端点信息
            self.get_logger().info(
                "  端点绑定: "
                f"move_action={self._resolve_move_group_endpoint(self.move_group_namespace, 'move_action')}, "
                f"plan_kinematic_path={self._resolve_move_group_endpoint(self.move_group_namespace, 'plan_kinematic_path')}, "
                f"execute_trajectory={self._resolve_move_group_endpoint(self.move_group_namespace, 'execute_trajectory')}, "
                f"check_state_validity={self._resolve_move_group_endpoint(self.move_group_namespace, 'check_state_validity')}"
            )

        except Exception as exc:
            self.get_logger().error(f"MoveIt初始化失败: {exc}")
            self.get_logger().error(traceback.format_exc())
            raise

    def _sync_state_validity_client(self):
        clients = getattr(self, "_state_validity_clients", {})
        client = clients.get(self.move_group_namespace)
        if client is None:
            client = self.create_client(
                GetStateValidity,
                self._resolve_move_group_endpoint(
                    self.move_group_namespace, "check_state_validity"
                ),
                callback_group=self.callback_group,
            )
            clients[self.move_group_namespace] = client
        self._state_validity_clients = clients
        self.state_validity_client = client

    def setup_ik_comparison(self):
        """创建 Fairino/KDL 原始 IK 服务客户端，不改变规划场景。"""
        self.ik_timeout = float(self.get_parameter("ik_timeout").value)
        self.fairino_ik_client = self.create_client(
            GetPositionIK,
            self._resolve_move_group_endpoint(
                self.move_group_namespaces["fairino"], "compute_ik"
            ),
            callback_group=self.callback_group,
        )
        self.kdl_ik_client = self.create_client(
            GetPositionIK,
            self._resolve_move_group_endpoint(
                self.move_group_namespaces["kdl"], "compute_ik"
            ),
            callback_group=self.callback_group,
        )

    # ═══════════════════════════════════════════════════════
    #  位姿构造与交互命令解析
    # ═══════════════════════════════════════════════════════

    def make_pose_from_xyzrpy(self, xyz: Tuple[float, float, float], rpy_deg) -> Pose:
        """根据 xyz 和 RPY 欧拉角（度）生成 Pose 消息。"""
        p = Pose()
        p.position.x = float(xyz[0])
        p.position.y = float(xyz[1])
        p.position.z = float(xyz[2])

        quat = self._pose_quat_from_rpy(rpy_deg)
        p.orientation.x = float(quat[0])
        p.orientation.y = float(quat[1])
        p.orientation.z = float(quat[2])
        p.orientation.w = float(quat[3])

        return p

    def _tty_input(self):
        """从 /dev/tty 读取一行，绕过 ros2 launch 的 stdin 重定向。"""
        with open("/dev/tty", "r") as tty:
            return tty.readline()

    @staticmethod
    def _normalize_command(raw: str) -> str:
        """标准化用户输入命令（去除下划线/连字符，映射到固定命令）。"""
        text = raw.strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(text.split())
        if text in ("go start", "gostart", "start"):
            return "go_start"
        if text in ("recover", "reset"):
            return "recover"
        return text

    @staticmethod
    def _normalize_planning_pipeline(pipeline: str) -> str:
        """标准化规划管线名称。"""
        return PlannerSwitch.normalize_pipeline(pipeline)

    @staticmethod
    def _normalize_planner_id(pipeline: str, algorithm: str) -> str:
        """标准化规划器算法名称，支持别名映射（仅 Fairino 管线）。"""
        algorithm_text = str(algorithm).strip()
        if not algorithm_text:
            return "birrt*" if PlannerSwitch.normalize_pipeline(pipeline) == "fairino" else algorithm_text
        return PlannerSwitch.normalize_planner(pipeline, algorithm_text)

    @staticmethod
    def _is_valid_planner_id(pipeline: str, algorithm: str) -> bool:
        """检查规划器 ID 是否有效。"""
        return PlannerSwitch.is_valid(pipeline, algorithm)

    def read_pose_or_command(self, prompt):
        """
        从终端读取用户输入，返回动作类型与数据。
        返回:
            ("pose", ((x, y, z), (rx, ry, rz)))
            ("go_start", None)
            ("recover", None)
            ("switch_ik", plugin_str)
            ("switch_planner", (pipeline_str, algorithm_str, raw_algorithm_str))
        """
        fallback_rpy = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        while rclpy.ok():
            sys.stderr.write(
                f"\n{'=' * 60}\n{prompt}\n"
                "支持输入:\n"
                "  1) x y z rx ry rz            例: 0.30 0.25 0.35 0 -180 0\n"
                "  2) x y z                      使用 target_rpy_deg 作为固定姿态\n"
                "  3) go start                   返回配置起点\n"
                "  4) recover                    重置 demo 场景\n"
                "  5) ik fairino / ik kdl         切换 IK 求解器\n"
                "  6) planner fairino mire_biait* / aapf_birrt* / birrt* / rrt / rrt* / informed_rrt* / prm\n"
                "     planner ompl RRTConnectFast\n"
                f"{'=' * 60}\n> "
            )
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            raw = raw.strip()

            # 按空格分词，优先检测 IK/planner 切换命令
            parts = raw.split()
            if len(parts) >= 2 and parts[0].lower() == "ik":
                plugin = parts[1].strip().lower()
                if plugin in ("fairino", "kdl"):
                    return ("switch_ik", plugin)
            if len(parts) >= 3 and parts[0].lower() == "planner":
                pipeline = self._normalize_planning_pipeline(parts[1])
                raw_algorithm = parts[2].strip()
                algorithm = self._normalize_planner_id(pipeline, raw_algorithm)
                return ("switch_planner", (pipeline, algorithm, raw_algorithm))

            # 标准化命令（go_start / recover）
            command = self._normalize_command(raw)
            if command in ("go_start", "recover"):
                return command, None

            # 尝试解析数字
            values = raw.replace(",", " ").split()
            if len(values) not in (3, 6):
                sys.stderr.write(
                    f"输入无效：请输入 3 或 6 个数字，或输入 go home/recover/ik/planner。"
                    f"当前收到 {len(values)} 个字段。\n"
                )
                sys.stderr.flush()
                continue

            try:
                pose_values = [float(v) for v in values]
                return "pose", self._parse_pose_values(pose_values, fallback_rpy)
            except ValueError:
                sys.stderr.write("输入包含非数字，请重新输入。\n")
                sys.stderr.flush()

        raise RuntimeError("rclpy shutdown")

    def ask_continue(self, prompt="继续规划测试? 输入 Y 继续，输入 N 结束: "):
        """询问用户是否继续下一轮。"""
        while rclpy.ok():
            sys.stderr.write(f"\n{prompt}")
            sys.stderr.flush()

            raw = self._tty_input()
            if not raw:
                raise RuntimeError("tty closed")

            choice = raw.strip().lower()
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False

            sys.stderr.write("请输入 Y 或 N。\n")
            sys.stderr.flush()

        return False

    # ═══════════════════════════════════════════════════════
    #  运动控制（位姿移动、关节移动、HOME）
    # ═══════════════════════════════════════════════════════

    def pose_to_pose_stamped(self, pose):
        """将 Pose 包装为 PoseStamped，附加时间戳和坐标系。"""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame_name
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose = pose
        return pose_stamped

    def _last_execution_error_code_value(self) -> str:
        """获取上一次执行的错误码字符串。"""
        error_code = self.moveit2_arm.get_last_execution_error_code()
        if error_code is None:
            return ""
        return str(error_code.val)

    def move_to_pose(self, target_pose, cartesian=False, action_name="移动"):
        """控制机械臂运动到目标位姿。"""
        target_pose_stamped = self.pose_to_pose_stamped(target_pose)

        try:
            self.get_logger().info(
                f"正在{action_name}: "
                f"pos=({target_pose.position.x:.3f}, "
                f"{target_pose.position.y:.3f}, "
                f"{target_pose.position.z:.3f}), "
                f"cartesian={cartesian}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )

            self.moveit2_arm.move_to_pose(
                pose=target_pose_stamped,
                cartesian=cartesian,
            )
            ok = self.moveit2_arm.wait_until_executed()

            if not ok:
                self.get_logger().error(
                    f"✗ {action_name}失败：执行未成功, error_code={self._last_execution_error_code_value()}"
                )
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            self.get_logger().error(traceback.format_exc())
            return False

    def move_to_joint(self, joint_positions, action_name="关节运动", accept_verified_timeout=False):
        """控制机器人运动到指定关节构型。"""
        try:
            self.get_logger().info(
                f"正在{action_name}: joints={[f'{j:.3f}' for j in joint_positions]}, "
                f"pipeline={self.moveit2_arm.pipeline_id}, "
                f"planner={self.moveit2_arm.planner_id}"
            )
            # 如果已在目标附近则跳过
            current_joints = self._current_joint_positions_ordered(timeout=0.5)
            if current_joints is not None and len(current_joints) == len(joint_positions) and all(
                error < 0.03
                for error in self._joint_position_errors(current_joints, joint_positions)
            ):
                self.get_logger().info(f"✓ {action_name}已在目标附近，跳过零位移执行")
                return True

            self.moveit2_arm.move_to_configuration(joint_positions)
            ok = self.moveit2_arm.wait_until_executed()

            if not ok:
                # 若允许通过实测关节状态验证超时
                if accept_verified_timeout and self._wait_until_joint_state_near(
                    joint_positions,
                    tol=0.03,
                    timeout=self.start_settle_timeout_s,
                    label="start after execution timeout",
                ):
                    self.get_logger().warn(
                        "start execution action timed out, but the measured joint state reached start"
                    )
                    return True
                self.get_logger().error(
                    f"✗ {action_name}失败, error_code={self._last_execution_error_code_value()}"
                )
                return False

            self.get_logger().info(f"✓ {action_name}完成")
            time.sleep(self.action_delay)
            return True

        except Exception as exc:
            self.get_logger().error(f"✗ {action_name}失败: {exc}")
            return False

    def _resolve_start_joint_state(self, require_collision_validation):
        """Return the configured joint-space start state."""
        joints = [float(value) for value in self.start_joints]
        if require_collision_validation and not self._is_joint_state_valid_for_benchmark(joints):
            raise RuntimeError("start_joints_state_invalid")
        self.start_joint_state = joints
        return self.start_joint_state

    def go_start(self):
        """Move to the configured joint-space start state."""
        joints = self._resolve_start_joint_state(self.scene_manager is not None)
        return self.move_to_joint(
            joints,
            action_name="返回起点",
            accept_verified_timeout=True,
        )

    def _ensure_start(self):
        joints = self._resolve_start_joint_state(require_collision_validation=True)
        for attempt in range(3):
            current = self._current_joint_positions_ordered(timeout=0.5)
            if current is not None and all(
                error < 0.03 for error in self._joint_position_errors(current, joints)
            ):
                return True, ""
            if self.go_start():
                return True, ""
            if attempt < 2:
                self.get_logger().warn(
                    f"start command unavailable; retrying ({attempt + 1}/3)"
                )
                time.sleep(1.0)
        return False, self._last_execution_error_code_value() or "start_reset_failed"

    # ═══════════════════════════════════════════════════════
    #  关节状态读取与等待
    # ═══════════════════════════════════════════════════════

    def _current_joint_positions_ordered(self, timeout=0.5) -> Optional[List[float]]:
        """在超时时间内获取按 joint_names 排序的当前关节位置。"""
        deadline = time.time() + max(0.05, float(timeout))
        while time.time() < deadline:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None:
                return ordered_positions
            time.sleep(0.02)

        self.get_logger().error(
            f"未能在 {timeout:.2f}s 内获取完整 joint state，joint_names={self.joint_names}"
        )
        return None

    def _wait_for_complete_joint_state(self, timeout, label):
        deadline = time.monotonic() + max(1.0, float(timeout))
        while rclpy.ok() and time.monotonic() < deadline:
            if self._ordered_joint_positions(self.moveit2_arm.joint_state) is not None:
                time.sleep(0.25)
                if self._ordered_joint_positions(self.moveit2_arm.joint_state) is not None:
                    self.get_logger().info(f"{label} joint state ready")
                    return True
            time.sleep(0.1)
        self.get_logger().error(f"Timed out waiting for complete joint state before {label}")
        return False

    def _ordered_joint_positions(self, joint_state) -> Optional[List[float]]:
        """从 JointState 消息中提取按 joint_names 排序的位置列表。"""
        if joint_state is None:
            return None
        names = list(joint_state.name) if hasattr(joint_state, "name") else []
        positions = list(joint_state.position) if hasattr(joint_state, "position") else []
        if not positions:
            return None
        # 有名称时按名称匹配
        if names and len(names) == len(positions):
            name_to_pos = {str(name): float(pos) for name, pos in zip(names, positions)}
            try:
                return [name_to_pos[joint_name] for joint_name in self.joint_names]
            except KeyError:
                return None
        # 无名称时假设顺序与 joint_names 一致
        if len(positions) >= len(self.joint_names):
            return [float(v) for v in positions[:len(self.joint_names)]]
        return None

    @staticmethod
    def _joint_position_errors(current_joints, target_joints) -> List[float]:
        """计算各关节角度误差（弧度），考虑环绕。"""
        return [
            abs(math.atan2(math.sin(float(current) - float(target)),
                           math.cos(float(current) - float(target))))
            for current, target in zip(current_joints, target_joints)
        ]

    def _wait_until_joint_state_near(self, target_joints, tol=0.05, timeout=8.0, label="target"):
        """轮询关节状态直到所有关节误差小于 tol 或超时。"""
        t0 = time.time()
        last_errors = None
        while time.time() - t0 < timeout:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is not None and len(ordered_positions) == len(target_joints):
                last_errors = self._joint_position_errors(ordered_positions, target_joints)
            if last_errors is not None and all(error < tol for error in last_errors):
                self.get_logger().info(
                    f"{label} joint convergence: elapsed_s={time.time() - t0:.3f} "
                    f"max_error_rad={max(last_errors):.5f} tol_rad={tol:.5f}"
                )
                return True
            time.sleep(0.05)
        if last_errors is not None:
            joint_errors = ", ".join(
                f"{joint_name}={error:.5f}"
                for joint_name, error in zip(self.joint_names, last_errors)
            )
            detail = f"max_error_rad={max(last_errors):.5f} errors=[{joint_errors}]"
        else:
            detail = "joint_state=unavailable_or_incomplete"
        self.get_logger().warn(
            f"Joint state did not converge to {label} within {timeout:.1f}s: "
            f"tol_rad={tol:.5f} {detail}"
        )
        return False

    # ═══════════════════════════════════════════════════════
    #  规划器与场景管理
    # ═══════════════════════════════════════════════════════

    def set_ik(self, plugin: str):
        """切换 IK/client 状态，不隐式修改规划管线。"""
        plugin = PlannerSwitch.normalize_ik(plugin)
        if plugin not in getattr(self, "moveit2_arms", {}):
            self.get_logger().error(f"无效 IK 插件: {plugin}，仅支持 fairino/kdl")
            return False

        self.ik_plugin = plugin
        self.moveit2_arm = self.moveit2_arms[plugin]
        self.move_group_namespace = self.move_group_namespaces[plugin]
        self._sync_state_validity_client()
        self.get_logger().info(
            f"IK/client 已切换: {plugin}, pipeline保持={self.moveit2_arm.pipeline_id}"
        )
        return True

    def set_planner(self, pipeline="fairino", algorithm="birrt*", raw_algorithm=None):
        """设置规划管线与算法。"""
        pipeline = self._normalize_planning_pipeline(pipeline)
        algorithm = self._normalize_planner_id(pipeline, algorithm)
        raw_algorithm = algorithm if raw_algorithm is None else str(raw_algorithm).strip()

        if not self._is_valid_planner_id(pipeline, algorithm):
            self.get_logger().error(
                f"无效 Fairino planner_id: raw='{raw_algorithm}', normalized='{algorithm}'；"
                "仅支持 mire_biait*, aapf_birrt*, birrt*, rrt, rrt*, informed_rrt*, prm"
            )
            return False

        arms = getattr(self, "moveit2_arms", None) or {"current": self.moveit2_arm}
        for arm in arms.values():
            arm.pipeline_id = pipeline
            arm.planner_id = algorithm
        self.get_logger().info(
            f"规划器已切换: pipeline={pipeline}, raw_algorithm={raw_algorithm}, "
            f"algorithm={algorithm}"
        )
        return True

    def _seed_robot_state(self):
        state = RobotState()
        joint_state = getattr(self.moveit2_arm, "joint_state", None)
        if joint_state is not None and joint_state.position:
            state.joint_state = joint_state
            return state
        if self.start_joint_state is not None:
            state.joint_state = JointState(
                name=list(self.joint_names), position=list(self.start_joint_state)
            )
        return state

    def _build_ik_request(self, pose):
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ee_frame_name
        request.ik_request.pose_stamped = self.pose_to_pose_stamped(pose)
        request.ik_request.avoid_collisions = False
        request.ik_request.timeout = Duration(
            sec=int(self.ik_timeout),
            nanosec=int((self.ik_timeout % 1.0) * 1e9),
        )
        request.ik_request.robot_state = self._seed_robot_state()
        return request

    @staticmethod
    def _ik_error_text(code):
        labels = {
            MoveItErrorCodes.SUCCESS: "SUCCESS",
            MoveItErrorCodes.FAILURE: "FAILURE",
            MoveItErrorCodes.PLANNING_FAILED: "PLANNING_FAILED",
            MoveItErrorCodes.TIMED_OUT: "TIMED_OUT",
            MoveItErrorCodes.NO_IK_SOLUTION: "NO_IK_SOLUTION",
        }
        return labels.get(code, f"UNKNOWN({code})")

    def _call_ik(self, label, client, pose):
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error(f"{label}: IK 服务不可用")
            return False, None
        started = time.perf_counter()
        future = client.call_async(self._build_ik_request(pose))
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        response = future.result()
        elapsed = time.perf_counter() - started
        if response is None:
            self.get_logger().error(f"{label}: IK 服务无响应")
            return False, None
        joint_map = dict(zip(response.solution.joint_state.name, response.solution.joint_state.position))
        joints = [joint_map.get(name) for name in self.joint_names]
        ok = response.error_code.val == MoveItErrorCodes.SUCCESS and all(value is not None for value in joints)
        self.get_logger().info(
            f"{label}: ok={ok}, code={self._ik_error_text(response.error_code.val)}, "
            f"time={elapsed:.4f}s"
        )
        if ok:
            self.get_logger().info(
                f"{label}: joints={[round(float(value), 6) for value in joints]}"
            )
        return ok, joints if ok else None

    def compare_ik(self, pose):
        fairino_ok, fairino_joints = self._call_ik("Fairino", self.fairino_ik_client, pose)
        kdl_ok, kdl_joints = self._call_ik("KDL", self.kdl_ik_client, pose)
        if fairino_ok and kdl_ok:
            distance = np.linalg.norm(np.array(fairino_joints) - np.array(kdl_joints))
            self.get_logger().info(f"Fairino/KDL |dq|={distance:.6f} rad")
        elif fairino_ok:
            self.get_logger().warn("Fairino 成功，KDL 失败")
        elif kdl_ok:
            self.get_logger().warn("Fairino 失败，KDL 成功")
        else:
            self.get_logger().error("Fairino 与 KDL 均失败")
        return fairino_joints if self.ik_plugin == "fairino" and fairino_ok else (
            kdl_joints if self.ik_plugin == "kdl" and kdl_ok else None
        )

    def report_tf_position_error(self, pose):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame_name, self.ee_frame_name, rclpy.time.Time()
            )
        except TransformException as exc:
            self.get_logger().warn(f"无法读取 TF 位置误差: {exc}")
            return
        actual = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        target = np.array([pose.position.x, pose.position.y, pose.position.z])
        self.get_logger().info(f"IK 执行后位置误差={np.linalg.norm(actual - target):.6f} m")

    def add_scene_obstacles(self):
        """向规划场景发布 YAML 中选定场景的障碍物。"""
        self.setup_scene()
        self.scene_manager.add_scene(self.active_obstacles)

    def clear_demo_collision_objects(self):
        """清除场景中所有障碍物。"""
        if self.scene_manager is not None:
            self.scene_manager.clear_scene(self.active_obstacles)

    def recover_demo_state(self):
        """
        重置 demo 到初始状态：
        1. 清除障碍物和末端轨迹
        2. 重置规划器到默认参数
        3. 重新添加 YAML 场景障碍物
        4. 机械臂回配置起点
        """
        self.get_logger().warn("执行 recover: 清除障碍物 → 重置规划器 → 重新加载 → 回起点")

        # 1. 清除障碍物和末端轨迹
        self.clear_demo_collision_objects()
        self.clear_ee_trace()

        # 2. 重置 IK 和规划器
        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)
        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)

        # 3. 重新加载障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_scene_obstacles()

        # 4. 在已发布场景中校验并返回配置起点
        self.go_start()

        self.get_logger().info("recover 完成")

    # ═══════════════════════════════════════════════════════
    # 可复现 benchmark（与交互节点共用 MoveIt、场景和规划器）
    # ═══════════════════════════════════════════════════════

    def _obstacle_signature(self):
        return obstacle_signature(self.active_obstacles)

    def _goal_set_spec(self, target_rpy_deg):
        return GoalSetSpec(
            scene_name=self.scene_name,
            goal_mode=self.benchmark_goal_mode,
            goal_seed=self.benchmark_goal_seed,
            ik_mode=self.ik_mode,
            obstacle_signature=self._obstacle_signature(),
            target_rpy_deg=tuple(float(value) for value in target_rpy_deg),
            repetitions=self.benchmark_repetitions,
            min_separation_m=self.benchmark_goal_min_separation_m,
            clearance_min_m=self.benchmark_goal_clearance_min_m,
            clearance_max_m=self.benchmark_goal_clearance_max_m,
            corridor_clearance_max_m=self.benchmark_goal_corridor_clearance_max_m,
            sampling_start_xyz=tuple(float(value) for value in self._start_tcp_xyz()),
            sampling_start_joints=tuple(float(value) for value in self.start_joints),
            start_id=self.start_id,
        )

    def _start_tcp_xyz(self):
        if self.start_xyz is None:
            if self.start_joint_state is None:
                raise RuntimeError("start_joint_state is not prepared")
            poses = self.moveit2_arm.compute_fk(
                joint_state=self.start_joint_state,
                fk_link_names=[self.ee_frame_name],
            )
            if not poses:
                raise RuntimeError("start_joints_fk_failed")
            pose = poses[0] if isinstance(poses, list) else poses
            position = pose.pose.position
            self.start_xyz = (float(position.x), float(position.y), float(position.z))
        return self.start_xyz

    def _benchmark_start_and_rpy(self):
        target_rpy = tuple(self._parse_float_list(self.get_parameter("target_rpy_deg").value))
        return self._start_tcp_xyz(), target_rpy

    def _joint_state_message(self, values, names=None):
        if hasattr(values, "joint_state"):
            values = values.joint_state
        if isinstance(values, (list, tuple)):
            positions, names = values, names or self.joint_names
        else:
            positions = getattr(values, "position", ())
            names = getattr(values, "name", ()) or names or self.joint_names
        positions = list(positions)
        names = list(names)
        if len(positions) < len(self.joint_names):
            return None
        if names and len(names) == len(positions):
            mapping = dict(zip(names, positions))
            if not all(name in mapping for name in self.joint_names):
                return None
            positions = [mapping[name] for name in self.joint_names]
        msg = JointState()
        msg.name, msg.position = list(self.joint_names), [float(v) for v in positions[:len(self.joint_names)]]
        return msg

    def _is_joint_state_valid_for_benchmark(self, joint_state, timeout=None):
        msg = self._joint_state_message(joint_state)
        timeout = self.benchmark_goal_state_validity_timeout_s if timeout is None else timeout
        if msg is None or not self.state_validity_client.wait_for_service(timeout_sec=max(0.1, float(timeout))):
            return False
        request = GetStateValidity.Request()
        request.group_name = self.group_name
        request.robot_state = RobotState(joint_state=msg)
        future = self.state_validity_client.call_async(request)
        deadline = time.monotonic() + max(0.1, float(timeout))
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                try:
                    return bool(future.result().valid)
                except Exception:
                    return False
            time.sleep(0.01)
        return False

    @staticmethod
    def _joint_trajectory_path_length(trajectory):
        return joint_trajectory_path_length(trajectory)

    def _plan_pose_from_start(self, target_pose):
        if self.start_joint_state is None:
            raise RuntimeError("start_joint_state is not prepared")
        future = self.moveit2_arm.plan_async(
            pose=self.pose_to_pose_stamped(target_pose), start_joint_state=self.start_joint_state, cartesian=False
        )
        if future is None:
            return {"success": False, "failure_code": "plan_request_failed", "core_planning_time_s": 0.0, "trajectory": None}
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        try:
            response = future.result().motion_plan_response
        except Exception:
            return {"success": False, "failure_code": "plan_response_failed", "core_planning_time_s": 0.0, "trajectory": None}
        trajectory = response.trajectory.joint_trajectory
        success = response.error_code.val == MoveItErrorCodes.SUCCESS and bool(trajectory.points)
        return {
            "success": success,
            "failure_code": "" if success else (
                "empty_trajectory" if response.error_code.val == MoveItErrorCodes.SUCCESS
                else "planning_timeout" if response.error_code.val == MoveItErrorCodes.TIMED_OUT
                else f"moveit_{response.error_code.val}"
            ),
            "core_planning_time_s": float(response.planning_time),
            "trajectory": trajectory if success else None,
        }

    def _publish_display_trajectory(self, trajectory):
        if trajectory is None or not trajectory.points:
            return
        display = DisplayTrajectory()
        display.trajectory_start.joint_state.name = list(self.joint_names)
        display.trajectory_start.joint_state.position = list(self.start_joint_state or [])
        display.trajectory.append(RobotTrajectory(joint_trajectory=trajectory))
        self.display_trajectory_pub.publish(display)

    def _execute_joint_trajectory(self, trajectory):
        return execute_joint_trajectory(
            self.moveit2_arm, trajectory, self._last_execution_error_code_value
        )

    def _benchmark_ik(self, point_xyz, goal_rpy):
        client = getattr(self, "fairino_ik_client", None)
        if client is None:
            result = self.moveit2_arm.compute_ik(
                position=point_xyz, quat_xyzw=self._pose_quat_from_rpy(goal_rpy),
                start_joint_state=self.start_joint_state, wait_for_server_timeout_sec=0.5,
            )
            return result, "ik_other" if result is None else ""
        if not client.wait_for_service(timeout_sec=0.5):
            return None, "ik_other"
        future = client.call_async(
            self._build_ik_request(self.make_pose_from_xyzrpy(point_xyz, goal_rpy))
        )
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        response = future.result()
        if response is None:
            return None, "ik_other"
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            reason = (
                "ik_geometry"
                if response.error_code.val == MoveItErrorCodes.NO_IK_SOLUTION
                else "ik_other"
            )
            return None, reason
        return response.solution.joint_state, ""

    def _benchmark_candidate_status(self, point_xyz, goal_rpy, start_xyz):
        metrics = adaptive_challenge_metrics(
            point_xyz,
            start_xyz,
            self.active_obstacles,
            self.benchmark_goal_corridor_clearance_max_m,
        )
        if (
            not metrics["accepted"]
            or not self.benchmark_goal_clearance_min_m
            <= metrics["endpoint_clearance_m"]
            <= self.benchmark_goal_clearance_max_m
        ):
            return False, "geometry"
        result, ik_reason = self._benchmark_ik(point_xyz, goal_rpy)
        if result is None or self._joint_state_message(result) is None:
            return False, ik_reason or "ik_other"
        return (True, "") if self._is_joint_state_valid_for_benchmark(result) else (False, "state")

    def _iter_benchmark_goals(self, count, start_xyz, goal_rpy):
        minimum, maximum = goal_bounds(self.active_obstacles)
        rejected = {
            "geometry": 0, "ik_geometry": 0, "ik_other": 0,
            "state": 0, "separation": 0,
        }
        goals = []
        endpoint_clearances = []
        candidates = iter_random_candidates(minimum, maximum, self.benchmark_goal_seed)
        attempts = 0
        self._benchmark_goal_sampling_report = None
        while len(goals) < count:
            attempts += 1
            point = next(candidates)
            ok, reason = self._benchmark_candidate_status(point, goal_rpy, start_xyz)
            if not ok:
                rejected[reason if reason in rejected else "state"] += 1
                continue
            if not goal_is_separated(point, goals, self.benchmark_goal_min_separation_m):
                rejected["separation"] += 1
                continue
            goal = (point, tuple(goal_rpy))
            goals.append(goal)
            endpoint_clearances.append(
                distance_to_obstacle_surface(point, self.active_obstacles)
            )
            self._benchmark_goal_sampling_report = build_goal_sampling_report(
                self.scene_name,
                self.benchmark_goal_seed,
                count,
                attempts,
                len(goals),
                rejected,
                endpoint_clearances,
            )
            yield goal
            if len(goals) == count:
                self.get_logger().info(
                    f"benchmark goals accepted={count}/{count} attempts={attempts} rejected={rejected}"
                )
                return

    def _run_manifest(self, paths, collection_manifest):
        with open(self.scene_config_file, "rb") as handle:
            scene_hash = hashlib.sha256(handle.read()).hexdigest()
        private_files = {
            "rrt": "rrt_params.yaml", "rrt_star": "rrt_star_params.yaml",
            "informed_rrt_star": "rrt_star_params.yaml",
            "birrt_star": "birrt_star_params.yaml",
            "aapf_birrt_star": "aapf_birrt_star_params.yaml",
            "mire_biait_star": "mire_biait_star_params.yaml", "prm": "prm_params.yaml",
        }
        planner_dir = os.path.join(get_package_share_directory("myrobot_planning_core"), "config")
        private_params = {}
        for planner, filename in private_files.items():
            with open(os.path.join(planner_dir, filename), encoding="utf-8") as handle:
                params = yaml.safe_load(handle)["fairino"]["algorithms"]
                config_key = {
                    "informed_rrt_star": "rrt_star",
                }.get(planner, planner)
                private_params[planner] = params[config_key]
        manifest = {
            "scene_name": self.scene_name,
            "scene_yaml_sha256": scene_hash,
            "goal_set_id": collection_manifest["goal_set_id"],
            "goal_set_relative_to_benchmark_output_dir": os.path.relpath(
                paths["directory"], self.benchmark_output_dir
            ),
            "goal_set_csv_sha256": collection_manifest["goal_set_csv_sha256"],
            "goal_set_signature_sha256": collection_manifest["goal_set_signature_sha256"],
            "goal_collection_format_version": collection_manifest["format_version"],
            "goal_collection_key": collection_manifest["collection_key"],
            "goal_set_sampling_identity": sampling_identity_payload(
                self._goal_set_spec(self._benchmark_start_and_rpy()[1])
            ),
            "start_id": self.start_id,
            "start_joints": list(self.start_joints),
            "start_tcp_xyz": list(self._start_tcp_xyz()),
            "start_joint_state": list(self.start_joint_state or []),
            "goal_root_mode": self.benchmark_goal_root_mode,
            "planner_id": self.default_planner_id,
            "planner_random_seed": self.planner_random_seed,
            "variant": self.benchmark_variant,
            "status": "prepared",
            "planning_scene_obstacle_padding_m": self.planning_scene_obstacle_padding_m,
            "scene_benchmark": dict(getattr(self, "scene_benchmark", {}) or {}),
            "effective_fairness_config": self.benchmark_effective_config,
            "algorithm_private_parameters": private_params,
        }

        signature_payload = dict(manifest)
        signature_payload.pop("status", None)
        manifest["run_signature_sha256"] = canonical_sha256(signature_payload)
        return manifest

    def _prepare_benchmark_artifacts(self, collection_manifest, paths):
        self._active_run_manifest = self._run_manifest(paths, collection_manifest)
        run_dir = prepare_benchmark_run(
            self.benchmark_output_dir,
            self.scene_name,
            collection_manifest["goal_set_id"],
            self.default_planner_id,
            self.planner_random_seed,
            self.benchmark_goal_root_mode,
            self._active_run_manifest,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            self.benchmark_variant,
        )
        finalize_run_manifest(run_dir, "running", self.benchmark_repetitions)
        return run_dir

    def _planner_stats_temp_path(self, prefix):
        return os.path.join(
            self.benchmark_output_dir,
            f".{prefix}_{benchmark_slug(self.default_planner_id)}"
            f"_seed{self.planner_random_seed}.csv",
        )

    def _sampling_stats_temp_path(self):
        return self._planner_stats_temp_path("aapf_sampling_stats")

    def _mire_stats_temp_path(self):
        return self._planner_stats_temp_path("mire_stats")

    def _planner_diagnostics_temp_path(self):
        return self._planner_stats_temp_path("planner_diagnostics")

    def _anytime_trace_temp_path(self):
        return self._planner_stats_temp_path("anytime_trace")

    def _root_diagnostics_temp_path(self):
        return self._planner_stats_temp_path("root_diagnostics")

    def _trajectory_paths_temp_path(self):
        return self._planner_stats_temp_path("trajectory_paths")

    def _benchmark_temp_paths(self):
        return (
            self._planner_diagnostics_temp_path(), self._sampling_stats_temp_path(),
            self._mire_stats_temp_path(), self._anytime_trace_temp_path(),
            self._root_diagnostics_temp_path(), self._trajectory_paths_temp_path(),
        )

    def _latest_planner_diagnostics(self):
        path = self._planner_diagnostics_temp_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            return rows[-1] if rows else {}
        except (OSError, csv.Error):
            return {}

    def _benchmark_result_row(self, index):
        manifest = self._active_run_manifest
        return {
            "scene_name": self.scene_name,
            "goal_set_id": manifest["goal_set_id"],
            "goal_set_signature_sha256": manifest["goal_set_signature_sha256"],
            "goal_index": index,
            "goal_root_mode": self.benchmark_goal_root_mode,
            "planner_id": self.default_planner_id,
            "planner_seed": self.planner_random_seed,
            "variant": self.benchmark_variant,
            "run_signature_sha256": manifest["run_signature_sha256"],
            "plan_success": "false",
            "failure_stage": "not_run",
            "failure_code": "not_run",
            "stop_reason": "not_run",
            "core_planning_time_s": 0.0,
            "first_solution_time_s": float("nan"),
            "first_solution_path_cost_rad": float("nan"),
            "joint_path_length_rad": float("nan"),
            "tcp_path_length_m": float("nan"),
            "joint_turn_total_variation_rad": float("nan"),
            "waypoint_count": 0,
            "planner_sample_attempts": 0,
            "planner_accepted_samples": 0,
            "planner_iterations": 0,
            "planner_work_units": 0,
            "planner_nodes": 0,
            "planner_edges": 0,
            "post_solution_sample_attempts": 0,
            "post_solution_budget_complete": "false",
            "collision_state_checks": 0,
            "collision_motion_checks": 0,
            "valid_motion_edges": 0,
            "invalid_motion_edges": 0,
            "ik_time_s": 0.0,
            "root_generation_time_s": 0.0,
            "search_time_s": 0.0,
            "final_validation_time_s": 0.0,
            "trajectory_construction_time_s": 0.0,
            "goal_root_count": 0,
            "selected_goal_root": -1,
        }

    def _apply_planner_diagnostics(self, row):
        diagnostics = self._latest_planner_diagnostics()
        if diagnostics:
            row.update(
                stop_reason=diagnostics.get("stop_reason", row["stop_reason"]),
                planner_sample_attempts=diagnostics.get("sample_attempts", 0),
                planner_accepted_samples=diagnostics.get("accepted_samples", 0),
                planner_iterations=diagnostics.get("iterations", 0),
                planner_nodes=diagnostics.get("num_nodes", 0),
                planner_work_units=diagnostics.get("work_units", 0),
                planner_edges=diagnostics.get("graph_edges", 0),
                post_solution_sample_attempts=diagnostics.get("post_solution_sample_attempts", 0),
                post_solution_budget_complete=diagnostics.get(
                    "post_solution_budget_complete", "false"),
                collision_state_checks=diagnostics.get("collision_state_checks", 0),
                collision_motion_checks=diagnostics.get("collision_motion_checks", 0),
                valid_motion_edges=diagnostics.get("valid_motion_edges", 0),
                invalid_motion_edges=diagnostics.get("invalid_motion_edges", 0),
                goal_root_count=diagnostics.get("goal_root_count", 0),
                selected_goal_root=diagnostics.get("selected_goal_root", -1),
                search_time_s=diagnostics.get("search_time_s", 0.0),
                ik_time_s=diagnostics.get("ik_time_s", 0.0),
                root_generation_time_s=diagnostics.get("root_generation_time_s", 0.0),
                final_validation_time_s=diagnostics.get("final_validation_time_s", 0.0),
                trajectory_construction_time_s=diagnostics.get("trajectory_construction_time_s", 0.0),
            )
            # Path-quality fields are defined only for successful attempts.
            # Keeping them NaN for failures is required for ITT/censoring analysis.
            if str(row.get("plan_success", "false")).lower() == "true":
                row.update(
                    first_solution_time_s=diagnostics.get("first_solution_time_s", float("nan")),
                    first_solution_path_cost_rad=diagnostics.get("first_solution_path_cost_rad", float("nan")),
                    tcp_path_length_m=diagnostics.get("optimized_tcp_path_length_m", float("nan")),
                    joint_turn_total_variation_rad=diagnostics.get(
                        "joint_turn_total_variation_rad", float("nan")),
                    waypoint_count=diagnostics.get("waypoint_count", 0),
                )

    @staticmethod
    def _read_temp_rows(path):
        if not os.path.isfile(path):
            return []
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _common_key(self, goal_index):
        manifest = self._active_run_manifest
        return {
            "scene_name": self.scene_name,
            "goal_set_id": manifest["goal_set_id"],
            "goal_set_signature_sha256": manifest["goal_set_signature_sha256"],
            "goal_index": goal_index,
            "goal_root_mode": self.benchmark_goal_root_mode,
            "planner_id": self.default_planner_id,
            "planner_seed": self.planner_random_seed,
            "variant": self.benchmark_variant,
            "run_signature_sha256": manifest["run_signature_sha256"],
        }

    def _collect_goal_sidecars(self, goal_index, result_row):
        key = self._common_key(goal_index)
        trace_source = self._read_temp_rows(self._anytime_trace_temp_path())
        by_checkpoint = {
            round(float(row["checkpoint_s"]), 9): row for row in trace_source
        }
        checkpoints = self.benchmark_effective_config.get("comparison", {}).get(
            "anytime_checkpoints_s", (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0)
        )
        last = None
        for checkpoint in checkpoints:
            source = by_checkpoint.get(round(float(checkpoint), 9), last)
            if source is not None:
                last = source
            first_solution_time = float(result_row.get("first_solution_time_s", 0.0) or 0.0)
            has_solution = (
                result_row.get("plan_success") == "true"
                and first_solution_time <= float(checkpoint)
            )
            source = source or {}
            source_has_solution = str(source.get("has_solution", "false")).lower() in ("true", "1", "yes")
            if has_solution and source_has_solution:
                incumbent = source.get("incumbent_joint_cost_rad", result_row.get("first_solution_path_cost_rad", "inf"))
            elif has_solution:
                incumbent = result_row.get("first_solution_path_cost_rad", "inf")
            else:
                incumbent = "inf"
            self._anytime_rows.append({
                **key,
                "checkpoint_s": checkpoint,
                "has_solution": str(has_solution).lower(),
                "incumbent_joint_cost_rad": incumbent,
                "cumulative_iterations": source.get(
                    "cumulative_iterations", result_row.get("planner_iterations", 0)),
                "cumulative_sample_attempts": source.get(
                    "cumulative_sample_attempts", result_row.get("planner_sample_attempts", 0)),
                "cumulative_accepted_samples": source.get(
                    "cumulative_accepted_samples", result_row.get("planner_accepted_samples", 0)),
                "cumulative_nodes": source.get(
                    "cumulative_nodes", result_row.get("planner_nodes", 0)),
                "cumulative_work_units": source.get(
                    "cumulative_work_units", result_row.get("planner_work_units", 0)),
            })

        roots = self._read_temp_rows(self._root_diagnostics_temp_path())
        accepted_root_count = int(float(result_row.get("goal_root_count", 0) or 0))
        if accepted_root_count <= 0:
            accepted_root_count = sum(
                str(root.get("passed_hard_filter", "true")).lower() in ("true", "1", "yes")
                for root in roots
            )
        if not roots:
            roots = [{
                "root_index": -1, "passed_hard_filter": "false",
                "filter_reason": result_row.get("failure_code", "no_valid_root"),
                "total_cost": "", "selected_final": "false",
                **{f"q{joint}": "" for joint in range(1, 7)},
            }]
        for root in roots:
            self._root_rows.append({
                **key,
                "root_index": root.get("root_index", -1),
                **{f"q{joint}": root.get(f"q{joint}", "") for joint in range(1, 7)},
                "passed_hard_filter": root.get("passed_hard_filter", "true"),
                "filter_reason": root.get("filter_reason", ""),
                "total_cost": root.get("total_cost", ""),
                "assigned_sample_attempts": root.get("assigned_sample_attempts", 0),
                "accepted_samples": root.get("accepted_samples", 0),
                "path_improvements": root.get("path_improvements", 0),
                "selected_final": root.get("selected_final", "false"),
            })
        result_row["goal_root_count"] = accepted_root_count
        selected = [row for row in roots if row.get("selected_final") == "true"]
        result_row["selected_goal_root"] = selected[0].get("root_index", -1) if selected else -1

        diagnostic_sources = [self._latest_planner_diagnostics()]
        diagnostic_sources += self._read_temp_rows(self._sampling_stats_temp_path())
        diagnostic_sources += self._read_temp_rows(self._mire_stats_temp_path())
        for diagnostic in diagnostic_sources:
            for name, value in diagnostic.items():
                if name in ("run_index", "planner_seed", "planner_id"):
                    continue
                if value is None or not str(value).strip():
                    continue
                try:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        continue
                    text = ""
                except (TypeError, ValueError):
                    numeric, text = float("nan"), str(value)
                self._algorithm_rows.append({
                    **key, "metric_name": name, "metric_value": numeric,
                    "metric_text": text, "ablation_variant": self.benchmark_variant,
                })
        if self.default_planner_id == "aapf_birrt*":
            for name in ("sample_tube", "sample_detour"):
                self._algorithm_rows.append({
                    **key, "metric_name": name, "metric_value": 0.0,
                    "metric_text": "pure_aapf", "ablation_variant": self.benchmark_variant,
                })
        if not diagnostic_sources[0]:
            self._algorithm_rows.append({
                **key, "metric_name": "diagnostics_available", "metric_value": 0.0,
                "metric_text": "false", "ablation_variant": self.benchmark_variant,
            })

        if result_row.get("plan_success") == "true":
            for waypoint in self._read_temp_rows(self._trajectory_paths_temp_path()):
                self._trajectory_rows.append({
                    **key,
                    "path_stage": waypoint.get("path_stage", "final"),
                    "waypoint_index": waypoint.get("waypoint_index", 0),
                    **{f"q{joint}": waypoint.get(f"q{joint}", "") for joint in range(1, 7)},
                    "tcp_x": waypoint.get("tcp_x", ""),
                    "tcp_y": waypoint.get("tcp_y", ""),
                    "tcp_z": waypoint.get("tcp_z", ""),
                    "selected_goal_root": result_row.get("selected_goal_root", -1),
                })

    def _write_benchmark_csvs(self, run_dir, result_rows):
        write_results(os.path.join(run_dir, "results.csv"), result_rows)
        write_csv_atomic(os.path.join(run_dir, "anytime_trace.csv"), self._anytime_rows, ANYTIME_FIELDS)
        write_csv_atomic(os.path.join(run_dir, "root_diagnostics.csv"), self._root_rows, ROOT_DIAGNOSTIC_FIELDS)
        write_csv_atomic(
            os.path.join(run_dir, "algorithm_diagnostics.csv"),
            self._algorithm_rows, ALGORITHM_DIAGNOSTIC_FIELDS)
        write_csv_atomic(os.path.join(run_dir, "trajectory_paths.csv"), self._trajectory_rows, TRAJECTORY_PATH_FIELDS)

    def _run_benchmark_goal(self, index, xyz, rpy):
        row = self._benchmark_result_row(index)
        for path in self._benchmark_temp_paths():
            if os.path.isfile(path):
                os.unlink(path)
        ready_for_goal = (
            not self.benchmark_executes_trajectory
            or index == 1
            or self._ensure_start()[0]
        )
        if not ready_for_goal:
            row["failure_stage"] = "start_reset"
            row["failure_code"] = "start_reset_failed"
            row["stop_reason"] = "precondition_failed"
            self._collect_goal_sidecars(index, row)
            return row

        result = self._plan_pose_from_start(self.make_pose_from_xyzrpy(xyz, rpy))
        row.update(
            plan_success=str(result["success"]).lower(),
            failure_stage="" if result["success"] else "planning",
            failure_code=result["failure_code"],
            stop_reason=("deadline_no_solution" if result["failure_code"] == "planning_timeout" else "completed"),
            core_planning_time_s=f"{result['core_planning_time_s']:.6f}",
        )
        self._apply_planner_diagnostics(row)
        trajectory = result["trajectory"]
        if trajectory is not None:
            row["joint_path_length_rad"] = (
                f"{self._joint_trajectory_path_length(trajectory):.6f}"
            )
            row["waypoint_count"] = len(trajectory.points)
            self._publish_display_trajectory(trajectory)
            if self.benchmark_executes_trajectory:
                ok, _code = self._execute_joint_trajectory(trajectory)
                if ok:
                    self.go_start()
        self._collect_goal_sidecars(index, row)
        return row

    def run_goal_collection(self):
        self.setup_scene()
        if not self._wait_for_complete_joint_state(self.benchmark_startup_joint_state_timeout_s, "goal collection start"):
            raise RuntimeError("runtime_not_ready: missing complete joint state")
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_scene_obstacles()
        self._resolve_start_joint_state(require_collision_validation=True)
        start_xyz, target_rpy = self._benchmark_start_and_rpy()
        spec = self._goal_set_spec(target_rpy)
        try:
            paths, goals, _manifest = load_goal_collection(self.goal_collection_dir, spec)
            self.get_logger().info(f"Reused goal collection: {paths['directory']} ({len(goals)} goals)")
            return
        except FileNotFoundError:
            pass
        goals = list(self._iter_benchmark_goals(self.benchmark_repetitions, start_xyz, target_rpy))
        if len(goals) != self.benchmark_repetitions or self._benchmark_goal_sampling_report is None:
            raise RuntimeError("goal collection ended before all valid goals were accepted")
        with open(self.scene_config_file, "rb") as handle:
            scene_sha = hashlib.sha256(handle.read()).hexdigest()
        paths, _goals, _manifest, reused = write_goal_collection(
            self.goal_collection_dir, spec, goals, scene_sha, self._benchmark_goal_sampling_report
        )
        self.get_logger().info(
            f"{'Reused' if reused else 'Collected'} goal collection: {paths['directory']}"
        )

    def run_benchmark(self):
        run_dir = None
        rows = []
        self._anytime_rows = []
        self._root_rows = []
        self._algorithm_rows = []
        self._trajectory_rows = []
        try:
            self.setup_scene()
            if not self._wait_for_complete_joint_state(self.benchmark_startup_joint_state_timeout_s, "benchmark start"):
                raise RuntimeError("runtime_not_ready: missing complete joint state")
            if self._as_bool(self.get_parameter("auto_add_obstacle").value): self.add_scene_obstacles()
            pre_start_ok, pre_start_error = self._ensure_start()
            if not pre_start_ok:
                raise RuntimeError(f"start_reset_failed: {pre_start_error}")
            for stale_stats in self._benchmark_temp_paths():
                if os.path.exists(stale_stats):
                    os.unlink(stale_stats)
            start_xyz, target_rpy = self._benchmark_start_and_rpy()
            goal_spec = self._goal_set_spec(target_rpy)
            paths, goals, collection_manifest = load_goal_collection(self.goal_collection_dir, goal_spec)
            run_dir = self._prepare_benchmark_artifacts(collection_manifest, paths)
            for index, (xyz, rpy) in enumerate(goals, 1):
                rows.append(self._run_benchmark_goal(index, xyz, rpy))
                self._write_benchmark_csvs(run_dir, rows)
            self._write_benchmark_csvs(run_dir, rows)
            finalize_run_manifest(run_dir, "complete", self.benchmark_repetitions)
            valid, reason = validate_complete_run(
                run_dir, self.benchmark_repetitions,
                tuple(self.benchmark_effective_config.get("comparison", {}).get(
                    "anytime_checkpoints_s", (0.1, 0.2, 0.5, 1, 2, 5, 10, 15))))
            if not valid:
                raise RuntimeError(f"run_integrity_failed: {reason}")
            write_benchmark_summary(run_dir)
        except Exception as exc:
            if run_dir:
                self._write_benchmark_csvs(run_dir, rows)
                finalize_run_manifest(
                    run_dir, "aborted", self.benchmark_repetitions, str(exc))
                write_benchmark_summary(run_dir)
            raise RuntimeError(str(exc))

    # ═══════════════════════════════════════════════════════
    #  demo 主循环
    # ═══════════════════════════════════════════════════════

    def select_mode(self):
        while rclpy.ok():
            sys.stderr.write(
                "\n选择功能:\n"
                "  1) 碰撞感知路径规划\n"
                "  2) Fairino/KDL IK 对比\n"
                "  q) 退出\n> "
            )
            sys.stderr.flush()
            choice = self._tty_input().strip().lower()
            if choice in ("1", "2", "q"):
                return choice
            sys.stderr.write("请输入 1、2 或 q。\n")
        return "q"

    def run_planning_mode(self):
        """交互式碰撞感知规划与场景管理。"""
        self.get_logger().info("=" * 70)
        self.get_logger().info("路径规划测试")
        self.get_logger().info("=" * 70)

        # 使用默认配置初始化规划器
        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)

        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)
        pipeline = self.moveit2_arm.pipeline_id
        algorithm = self.moveit2_arm.planner_id
        self.get_logger().info(
            f"配置: IK/client={ik_plugin}, pipeline={pipeline}, planner={algorithm}, "
            f"scene={self.scene_name}"
        )

        self.setup_scene()
        # 按需添加障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_scene_obstacles()

        # 可选：demo 前在已发布场景中先回配置起点
        if self.go_start_before_demo:
            if not self.go_start():
                self.get_logger().error("回起点失败，终止 demo")
                return
        else:
            self.get_logger().info("go_start_before_demo=false，启动后保持当前机械臂初始状态")

        while rclpy.ok():
            # 读取用户输入（目标位姿或命令）
            action, data = self.read_pose_or_command(
                "输入终点 pose: x y z [rx ry rz]，或输入 ik/planner/go start/recover"
            )

            if action == "go_start":
                self.go_start()
                if not self.ask_continue():
                    break
                continue

            if action == "recover":
                self.recover_demo_state()
                if not self.ask_continue():
                    break
                continue

            if action == "switch_ik":
                self.set_ik(data)
                continue

            if action == "switch_planner":
                pl_pipeline, pl_algorithm, pl_algorithm_raw = data
                if self.set_planner(pl_pipeline, pl_algorithm, pl_algorithm_raw):
                    pipeline = self.moveit2_arm.pipeline_id
                    algorithm = self.moveit2_arm.planner_id
                continue

            # action == "pose"：规划并运动到目标位姿
            goal_xyz, goal_rpy = data
            goal_pose = self.make_pose_from_xyzrpy(goal_xyz, goal_rpy)
            self.get_logger().info(
                f"终点: xyz=({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f}), "
                f"rpy_deg=({goal_rpy[0]:.1f}, {goal_rpy[1]:.1f}, {goal_rpy[2]:.1f})"
            )

            t0 = time.time()
            ok = self.move_to_pose(
                goal_pose,
                cartesian=False,
                action_name=f"{pipeline}/{algorithm} 当前位姿 -> 终点",
            )
            dt = time.time() - t0

            if ok:
                self.get_logger().info(f"终点执行成功，耗时={dt:.3f}s")
            else:
                self.get_logger().error(f"终点执行失败，耗时={dt:.3f}s")

            if not self.ask_continue():
                break

        # 清除障碍物（如果参数要求）
        if self._as_bool(self.get_parameter("remove_obstacle_after_demo").value):
            self.clear_demo_collision_objects()

        self.get_logger().info("路径规划 demo 结束")

    def run_ik_comparison_mode(self):
        """比较原始 IK 服务结果后直接按选中解执行，不加载规划场景。"""
        self.get_logger().info("Fairino/KDL IK 对比测试")
        self.go_start()
        while rclpy.ok():
            action, data = self.read_pose_or_command(
                "输入 IK 目标 pose: x y z [rx ry rz]，或输入 ik/planner/go start"
            )
            if action == "go_start":
                self.go_start()
            elif action == "switch_ik":
                self.set_ik(data)
            elif action == "switch_planner":
                self.set_planner(*data)
            elif action == "recover":
                self.get_logger().warn("IK 对比模式不管理场景；请使用 go start。")
            else:
                xyz, rpy = data
                pose = self.make_pose_from_xyzrpy(xyz, rpy)
                joints = self.compare_ik(pose)
                if joints is not None and self.move_to_joint(joints, "IK 解执行"):
                    self.report_tf_position_error(pose)
            if not self.ask_continue("继续 IK 对比? 输入 Y 继续，输入 N 返回菜单: "):
                return

    def run_demo(self):
        """从终端菜单选择路径规划或 IK 对比。"""
        if self.run_mode == "goal_collection":
            self.run_goal_collection()
            return
        if self.run_mode != "interactive":
            self.run_benchmark()
            return
        while rclpy.ok():
            mode = self.select_mode()
            if mode == "q":
                return
            if mode == "1":
                self.run_planning_mode()
            else:
                self.run_ik_comparison_mode()


def main(args=None):
    """节点入口：初始化 ROS，创建节点并启动交互式 demo。"""
    rclpy.init(args=args)

    node = MotionPlanningNodeSim()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # 后台线程处理回调
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        time.sleep(3.0)  # 等待节点完全就绪
        node.get_logger().info("开始执行任务...")
        node.run_demo()
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    except RuntimeError as exc:
        node.get_logger().error(f"任务失败: {exc}")
        node.get_logger().error(traceback.format_exc())
        exit_code = 2
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
