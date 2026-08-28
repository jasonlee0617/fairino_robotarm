#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# 交互式仿真规划与 IK 对比节点
#
# 提供交互式终端控制，支持：
#   - 输入目标位姿 (x y z 或 x y z rx ry rz) 进行规划与运动
#   - 切换 IK 求解器 (fairino / kdl)
#   - 切换规划算法 (birrt*, rrt*, aapf_birrt*, tube_birrt* 等)
#   - 返回 HOME、重置场景 (recover)
#   - 末端轨迹实时可视化 (RViz Marker)
#
# 依赖：
#   - pymoveit2 (MoveIt2 Python 接口)
#   - pathplanning_scene_tools (场景加载管理)
#   - scipy (RPY→四元数转换)
# ---------------------------------------------------------------------------

import csv
import hashlib
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
from pathplanning_scene_tools import SceneEnvironmentManager, SceneLoader
from myrobot_common.planning.motion_executor import PlannerSwitch
from planning_benchmark import (
    obstacle_attr, obstacle_center, obstacle_half_extents, select_farthest_goals,
    write_results, write_summary,
)
from planning_motion import execute_joint_trajectory, joint_trajectory_path_length
from planning_trace import append_trace_point

import tf2_ros
from tf2_ros import TransformException


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
        self.declare_parameter("home_joints", "-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0")
        self.declare_parameter("home_settle_timeout_s", 6.0)   # HOME 归位确认超时
        self.declare_parameter("ik_timeout", 3.0)

        # 规划参数
        self.declare_parameter("default_pipeline_id", "fairino")
        self.declare_parameter("default_planner_id", "birrt*")
        self.declare_parameter("target_rpy_deg", "0,-180,0")  # 默认末端姿态（RPY 度）
        self.declare_parameter("go_home_before_demo", False)

        # 场景与障碍物参数
        self.declare_parameter("auto_add_obstacle", True)
        self.declare_parameter("remove_obstacle_after_demo", True)
        self.declare_parameter("obstacle_name", "birrt_test_obstacle")
        self.declare_parameter("obstacle_position", "0.35,0.05,0.28")
        self.declare_parameter("obstacle_size", "0.18,0.45,0.35")
        self.declare_parameter("obstacle_boxes", "")
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
        self.declare_parameter("benchmark_output_dir", "")
        self.declare_parameter("benchmark_repetitions", 20)
        self.declare_parameter("benchmark_case_label", "")
        self.declare_parameter("benchmark_startup_joint_state_timeout_s", 90.0)
        self.declare_parameter("benchmark_goal_mode", "adaptive_obstacle_challenge_region")
        self.declare_parameter("benchmark_goal_seed", 17)
        self.declare_parameter("planner_random_seed", 7)
        self.declare_parameter("benchmark_goal_clearance_min_m", 0.06)
        self.declare_parameter("benchmark_goal_clearance_max_m", 0.14)
        self.declare_parameter("benchmark_goal_corridor_clearance_max_m", 0.10)
        self.declare_parameter("benchmark_goal_min_separation_m", 0.04)
        self.declare_parameter("benchmark_goal_candidate_count", 4096)
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
    def _csv_safe(value) -> str:
        return str(value).replace("\n", " ").replace(",", ";").strip()

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
        self.home_joints = self._parse_float_list(self.get_parameter("home_joints").value)

        self.default_pipeline_id = str(self.get_parameter("default_pipeline_id").value)
        self.default_planner_id = str(self.get_parameter("default_planner_id").value)
        self.default_planning_client = PlannerSwitch.normalize_ik(
            str(self.get_parameter("planning_client").value)
        )
        self.go_home_before_demo = self._as_bool(self.get_parameter("go_home_before_demo").value)
        self.home_settle_timeout_s = max(
            0.5, float(self.get_parameter("home_settle_timeout_s").value)
        )

        self.default_obstacle_name = str(self.get_parameter("obstacle_name").value)
        self.default_obstacle_position = tuple(
            self._parse_float_list(self.get_parameter("obstacle_position").value)
        )
        self.default_obstacle_size = tuple(
            self._parse_float_list(self.get_parameter("obstacle_size").value)
        )
        self.obstacle_boxes = SceneLoader.parse_obstacle_boxes(
            self.get_parameter("obstacle_boxes").value)
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
        if self.run_mode not in ("interactive", "benchmark_execution", "benchmark_algorithm"):
            raise ValueError(
                "run_mode must be interactive, benchmark_execution, or benchmark_algorithm"
            )
        self.benchmark_output_dir = self._resolve_benchmark_output_dir(
            self.get_parameter("benchmark_output_dir").value
        )
        if self.run_mode != "interactive":
            self.get_logger().info(
                f"Benchmark archive root: {self.benchmark_output_dir}"
            )
        self.benchmark_repetitions = max(1, int(self.get_parameter("benchmark_repetitions").value))
        self.benchmark_case_label = str(self.get_parameter("benchmark_case_label").value).strip()
        self.benchmark_startup_joint_state_timeout_s = max(
            1.0, float(self.get_parameter("benchmark_startup_joint_state_timeout_s").value)
        )
        self.benchmark_goal_mode = self._normalize_benchmark_goal_mode(
            self.get_parameter("benchmark_goal_mode").value
        )
        self.benchmark_goal_seed = int(self.get_parameter("benchmark_goal_seed").value)
        self.planner_random_seed = int(self.get_parameter("planner_random_seed").value)
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
        self.benchmark_goal_candidate_count = max(
            self.benchmark_repetitions,
            int(self.get_parameter("benchmark_goal_candidate_count").value),
        )
        self.benchmark_goal_state_validity_timeout_s = max(
            0.1, float(self.get_parameter("benchmark_goal_state_validity_timeout_s").value)
        )
        self.benchmark_executes_trajectory = self.run_mode == "benchmark_execution"

        # 基本校验
        if len(self.joint_names) != len(self.home_joints):
            raise ValueError("joint_names 与 home_joints 长度必须一致")
        if len(self.default_obstacle_position) != 3:
            raise ValueError("obstacle_position 必须包含 3 个数值")
        if len(self.default_obstacle_size) != 3:
            raise ValueError("obstacle_size 必须包含 3 个数值")

        # 运动间默认延迟（秒）
        self.action_delay = 1.0

        self.scene_manager = None
        self.active_obstacles = []

    def setup_scene(self):
        """仅在路径规划模式加载并发布场景。"""
        if self.scene_manager is not None:
            return
        self.scene_manager = SceneEnvironmentManager(
            node=self,
            base_frame_name=self.base_frame_name,
            scene_name=self.scene_name,
            scene_config_file=self.scene_config_file,
            scene_assets_dir=self.scene_assets_dir,
            sim_world=self.sim_world,
            obstacle_marker_topic=self.obstacle_marker_topic,
            publish_planning_scene=self.publish_planning_scene,
            publish_obstacle_markers=self.publish_obstacle_markers,
            spawn_sim_scene_models=self.spawn_sim_scene_models,
            planning_scene_obstacle_padding_m=self.planning_scene_obstacle_padding_m,
        )
        # 加载当前场景的障碍物列表
        self.active_obstacles = self.scene_manager.load_scene(
            self.obstacle_boxes,
            self.default_obstacle_name,
            self.default_obstacle_position,
            self.default_obstacle_size,
        )
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
        arm.allowed_planning_time = 15.0
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
            import traceback
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

    def make_pose_from_xyz(self, xyz: Tuple[float, float, float]) -> Pose:
        """使用默认 target_rpy_deg 姿态生成 Pose。"""
        rpy_deg = self._parse_float_list(self.get_parameter("target_rpy_deg").value)
        return self.make_pose_from_xyzrpy(xyz, rpy_deg)

    def _tty_input(self):
        """从 /dev/tty 读取一行，绕过 ros2 launch 的 stdin 重定向。"""
        with open("/dev/tty", "r") as tty:
            return tty.readline()

    @staticmethod
    def _normalize_command(raw: str) -> str:
        """标准化用户输入命令（去除下划线/连字符，映射到固定命令）。"""
        text = raw.strip().lower().replace("_", " ").replace("-", " ")
        text = " ".join(text.split())
        if text in ("go home", "gohome", "home"):
            return "go_home"
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
            ("go_home", None)
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
                "  3) go home                    返回 HOME\n"
                "  4) recover                    重置 demo 场景\n"
                "  5) ik fairino / ik kdl         切换 IK 求解器\n"
                "  6) planner fairino tube_birrt*  / birrt* / rrt* / aapf_birrt*\n"
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

            # 标准化命令（go_home / recover）
            command = self._normalize_command(raw)
            if command in ("go_home", "recover"):
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
            import traceback
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
                    timeout=self.home_settle_timeout_s,
                    label="HOME after execution timeout",
                ):
                    self.get_logger().warn(
                        "HOME execution action timed out, but the measured joint state reached HOME"
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

    def go_home(self):
        """返回 HOME 构型（允许超时后通过关节状态验证）。"""
        return self.move_to_joint(
            self.home_joints,
            action_name="返回HOME",
            accept_verified_timeout=True,
        )

    def _ensure_home(self):
        current = self._current_joint_positions_ordered(timeout=0.5)
        if current is not None and all(
            error < 0.03 for error in self._joint_position_errors(current, self.home_joints)
        ):
            return True, ""
        return (True, "") if self.go_home() else (
            False, self._last_execution_error_code_value() or "home_reset_failed"
        )

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

    def _wait_until_joint_state_stable(
        self,
        max_delta=0.005,
        stable_samples=4,
        sample_period_s=0.05,
        timeout=1.5,
        label="joint_state",
    ):
        """等待关节状态连续多次采样变化小于阈值（稳定）。"""
        t0 = time.time()
        prev_positions = None
        stable_count = 0
        last_delta = None
        while time.time() - t0 < timeout:
            ordered_positions = self._ordered_joint_positions(self.moveit2_arm.joint_state)
            if ordered_positions is None:
                time.sleep(sample_period_s)
                continue
            if prev_positions is not None and len(prev_positions) == len(ordered_positions):
                deltas = self._joint_position_errors(ordered_positions, prev_positions)
                last_delta = max(deltas) if deltas else 0.0
                if last_delta <= max_delta:
                    stable_count += 1
                    if stable_count >= stable_samples:
                        self.get_logger().info(
                            f"{label} joint stability: elapsed_s={time.time() - t0:.3f} "
                            f"max_delta_rad={last_delta:.5f} stable_samples={stable_count}"
                        )
                        return True
                else:
                    stable_count = 0
            prev_positions = ordered_positions
            time.sleep(sample_period_s)
        detail = (
            f"max_delta_rad={last_delta:.5f}" if last_delta is not None
            else "joint_state=unavailable_or_incomplete"
        )
        self.get_logger().warn(
            f"Joint state did not stabilize for {label} within {timeout:.1f}s: "
            f"max_delta_tol_rad={max_delta:.5f} {detail}"
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
                "仅支持 aapf_birrt*, tube_birrt*, birrt*, rrt*"
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
        state.joint_state = JointState(
            name=list(self.joint_names), position=list(self.home_joints)
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

    def add_default_obstacle(self):
        """向规划场景添加默认障碍物。"""
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
        2. 机械臂回 HOME
        3. 重置规划器到默认参数
        4. 重新添加默认障碍物
        """
        self.get_logger().warn("执行 recover: 回 HOME → 清除障碍物 → 重置规划器 → 重新加载")

        # 1. 清除障碍物和末端轨迹
        self.clear_demo_collision_objects()
        self.clear_ee_trace()

        # 2. 先回 HOME
        self.go_home()

        # 3. 重置 IK 和规划器
        ik_plugin = str(self.get_parameter("planning_client").value).strip().lower()
        pipeline = str(self.get_parameter("default_pipeline_id").value)
        algorithm = str(self.get_parameter("default_planner_id").value)
        self.set_ik(ik_plugin)
        self.set_planner(pipeline, algorithm)

        # 4. 重新加载障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        self.get_logger().info("recover 完成")

    # ═══════════════════════════════════════════════════════
    # 可复现 benchmark（与交互节点共用 MoveIt、场景和规划器）
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _obstacle_attr(obstacle, key, default=None):
        return obstacle_attr(obstacle, key, default)

    def _obstacle_center(self, obstacle):
        return obstacle_center(obstacle)

    def _obstacle_half_extents(self, obstacle):
        return obstacle_half_extents(obstacle)

    def _obstacle_signature(self):
        parts = []
        for obstacle in self.active_obstacles:
            parts.append({
                "name": str(self._obstacle_attr(obstacle, "name", "")),
                "shape": str(self._obstacle_attr(obstacle, "shape", "box")),
                "position": [round(v, 6) for v in self._obstacle_center(obstacle)],
                "half_extents": [round(v, 6) for v in self._obstacle_half_extents(obstacle)],
            })
        return hashlib.sha256(repr(sorted(parts, key=lambda item: item["name"])).encode()).hexdigest()

    def _distance_to_obstacle_surface(self, point_xyz, obstacles):
        point = np.asarray(point_xyz, dtype=float)
        distances = []
        for obstacle in obstacles:
            center = np.asarray(self._obstacle_center(obstacle), dtype=float)
            shape = str(self._obstacle_attr(obstacle, "shape", "box")).lower()
            if shape == "box":
                distances.append(float(np.linalg.norm(np.maximum(np.abs(point - center) - self._obstacle_half_extents(obstacle), 0.0))))
            elif shape == "cylinder":
                radius, _, half_height = self._obstacle_half_extents(obstacle)
                radial = max(0.0, float(np.linalg.norm(point[:2] - center[:2])) - radius)
                vertical = max(0.0, abs(point[2] - center[2]) - half_height)
                distances.append(float(np.hypot(radial, vertical)))
            else:
                distances.append(max(0.0, float(np.linalg.norm(point - center)) - self._obstacle_half_extents(obstacle)[0]))
        return min(distances) if distances else float("inf")

    def _adaptive_challenge_metrics(self, point_xyz, start_xyz):
        point = np.asarray(point_xyz, dtype=float)
        centers = [np.asarray(self._obstacle_center(item), dtype=float) for item in self.active_obstacles]
        angles = sorted(math.atan2(center[1] - point[1], center[0] - point[0]) for center in centers)
        if len(angles) >= 2:
            gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
            gaps.append(2.0 * math.pi - angles[-1] + angles[0])
            angular_coverage = math.degrees(2.0 * math.pi - max(gaps))
        else:
            angular_coverage = 0.0
        vertical = sum(abs(center[2] - point[2]) > 0.03 for center in centers)
        clearance = self._distance_to_obstacle_surface(point_xyz, self.active_obstacles)
        corridor = min(
            self._distance_to_obstacle_surface(
                point * (1.0 - alpha) + np.asarray(start_xyz) * alpha, self.active_obstacles
            ) for alpha in (0.25, 0.5, 0.75)
        )
        accepted = len(centers) >= 3 and vertical >= 2 and angular_coverage >= 180.0 and corridor <= self.benchmark_goal_corridor_clearance_max_m
        return {
            "accepted": accepted, "inside_obstacle_hull": angular_coverage >= 180.0,
            "surrounding_obstacle_count": len(centers), "vertical_obstacle_count": vertical,
            "angular_coverage_deg": angular_coverage, "corridor_min_clearance_m": corridor,
            "endpoint_clearance_m": clearance,
        }

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

    def _plan_pose_from_home(self, target_pose):
        future = self.moveit2_arm.plan_async(
            pose=self.pose_to_pose_stamped(target_pose), start_joint_state=self.home_joints, cartesian=False
        )
        if future is None:
            return {"success": False, "error_code": "plan_future_unavailable", "core_planning_time_s": 0.0, "trajectory": None}
        while rclpy.ok() and not future.done():
            time.sleep(0.01)
        try:
            response = future.result().motion_plan_response
        except Exception:
            return {"success": False, "error_code": "plan_exception", "core_planning_time_s": 0.0, "trajectory": None}
        trajectory = response.trajectory.joint_trajectory
        success = response.error_code.val == MoveItErrorCodes.SUCCESS and bool(trajectory.points)
        return {
            "success": success, "error_code": "" if success else str(response.error_code.val),
            "core_planning_time_s": float(response.planning_time),
            "trajectory": trajectory if success else None,
        }

    def _publish_display_trajectory(self, trajectory):
        if trajectory is None or not trajectory.points:
            return
        display = DisplayTrajectory()
        display.trajectory_start.joint_state.name = list(self.joint_names)
        display.trajectory_start.joint_state.position = list(self.home_joints)
        display.trajectory.append(RobotTrajectory(joint_trajectory=trajectory))
        self.display_trajectory_pub.publish(display)

    def _execute_joint_trajectory(self, trajectory):
        return execute_joint_trajectory(
            self.moveit2_arm, trajectory, self._last_execution_error_code_value
        )

    def _benchmark_candidate_status(self, point_xyz, goal_rpy, start_xyz):
        metrics = self._adaptive_challenge_metrics(point_xyz, start_xyz)
        if not metrics["accepted"] or not self.benchmark_goal_clearance_min_m <= metrics["endpoint_clearance_m"] <= self.benchmark_goal_clearance_max_m:
            return False, "geometry"
        result = self.moveit2_arm.compute_ik(
            position=point_xyz, quat_xyzw=self._pose_quat_from_rpy(goal_rpy),
            start_joint_state=self.home_joints, wait_for_server_timeout_sec=0.5,
        )
        if result is None or self._joint_state_message(result) is None:
            return False, "ik"
        return (True, "") if self._is_joint_state_valid_for_benchmark(result) else (False, "state")

    def _goal_is_valid_for_benchmark(self, point_xyz, goal_rpy, start_xyz, existing_goals):
        ok, _reason = self._benchmark_candidate_status(point_xyz, goal_rpy, start_xyz)
        return ok and not any(
            np.linalg.norm(np.asarray(point_xyz) - np.asarray(other[0])) < self.benchmark_goal_min_separation_m
            for other in existing_goals
        )

    def _goal_bounds(self):
        centers = np.asarray([self._obstacle_center(item) for item in self.active_obstacles], dtype=float)
        extents = np.asarray([self._obstacle_half_extents(item) for item in self.active_obstacles], dtype=float)
        if not len(centers):
            raise ValueError("benchmark scene has no obstacles")
        return np.min(centers - extents, axis=0), np.max(centers + extents, axis=0)

    def _generate_benchmark_goals(self, count, start_xyz, goal_rpy):
        minimum, maximum = self._goal_bounds()
        goals, diagnostics = select_farthest_goals(
            minimum, maximum, count, self.benchmark_goal_candidate_count,
            self.benchmark_goal_seed, self.benchmark_goal_min_separation_m,
            lambda point: self._benchmark_candidate_status(point, goal_rpy, start_xyz),
        )
        self.get_logger().info(f"benchmark candidate diagnostics: {diagnostics}")
        return [(point, tuple(goal_rpy)) for point in goals]

    def _write_generated_goals_csv(self, goals, path, start_xyz):
        fields = ["scene_name", "goal_mode", "goal_seed", "obstacle_signature", "goal_index", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"]
        with open(path + ".tmp", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, (xyz, rpy) in enumerate(goals, 1):
                writer.writerow(dict(zip(fields, [self.scene_name, self.benchmark_goal_mode, self.benchmark_goal_seed, self._obstacle_signature(), index, *xyz, *rpy])))
        os.replace(path + ".tmp", path)

    def _read_generated_goals_csv(self, path, start_xyz, expected_goal_rpy):
        goals = []
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["scene_name"] != self.scene_name or row["goal_mode"] != self.benchmark_goal_mode or int(row["goal_seed"]) != self.benchmark_goal_seed:
                    raise ValueError("goal 文件与当前 benchmark 条件不一致")
                if row["obstacle_signature"] != self._obstacle_signature():
                    raise ValueError("goal 文件的障碍物布局签名与当前场景不一致")
                xyz = tuple(float(row[key]) for key in ("x", "y", "z"))
                rpy = tuple(float(row[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg"))
                if not np.allclose(rpy, expected_goal_rpy, atol=1e-6) or not self._goal_is_valid_for_benchmark(xyz, rpy, start_xyz, goals):
                    raise ValueError("goal 文件不再满足当前有效性约束")
                goals.append((xyz, rpy))
        if len(goals) != self.benchmark_repetitions:
            raise ValueError("goal 数量与 benchmark_repetitions 不一致")
        return goals

    def _benchmark_config(self):
        scene_hash = hashlib.sha256(open(self.scene_config_file, "rb").read()).hexdigest()
        return {
            "case_label": self.benchmark_case_label or self.scene_name, "scene_name": self.scene_name,
            "scene_yaml_sha256": scene_hash, "goal_mode": self.benchmark_goal_mode,
            "goal_seed": self.benchmark_goal_seed, "repetitions": self.benchmark_repetitions,
            "target_rpy_deg": self._parse_float_list(self.get_parameter("target_rpy_deg").value),
            "goal_clearance_min_m": self.benchmark_goal_clearance_min_m,
            "goal_clearance_max_m": self.benchmark_goal_clearance_max_m,
            "goal_corridor_clearance_max_m": self.benchmark_goal_corridor_clearance_max_m,
            "goal_min_separation_m": self.benchmark_goal_min_separation_m,
            "planning_scene_obstacle_padding_m": self.planning_scene_obstacle_padding_m,
        }

    @staticmethod
    def _write_yaml(path, content):
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            yaml.safe_dump(content, handle, allow_unicode=True, sort_keys=True)
        os.replace(path + ".tmp", path)

    @staticmethod
    def _benchmark_run_dirs(case_dir):
        return sorted(
            entry.path for entry in os.scandir(case_dir)
            if entry.is_dir() and not entry.name.startswith(".")
        )

    @classmethod
    def _migrate_legacy_root_artifacts(cls, case_dir):
        """Move the former root snapshots into their sole run directory."""
        legacy_names = ("benchmark_config.yaml", "generated_goals.csv")
        legacy_paths = [os.path.join(case_dir, name) for name in legacy_names]
        if not any(os.path.exists(path) for path in legacy_paths):
            return
        run_dirs = cls._benchmark_run_dirs(case_dir)
        if len(run_dirs) != 1:
            raise RuntimeError(
                "legacy benchmark root artifacts require exactly one run directory"
            )
        run_dir = run_dirs[0]
        for name, source in zip(legacy_names, legacy_paths):
            if not os.path.exists(source):
                continue
            target = os.path.join(run_dir, name)
            if os.path.exists(target):
                with open(source, "rb") as source_handle, open(target, "rb") as target_handle:
                    if source_handle.read() != target_handle.read():
                        raise RuntimeError(f"legacy benchmark artifact conflicts with {target}")
                os.unlink(source)
            else:
                os.replace(source, target)

    def _prepare_benchmark_artifacts(self):
        if not self.benchmark_output_dir:
            raise RuntimeError("benchmark_output_dir is required")
        case_dir = os.path.abspath(self.benchmark_output_dir)
        os.makedirs(case_dir, exist_ok=True)
        self._migrate_legacy_root_artifacts(case_dir)
        config = self._benchmark_config()
        for run_dir in self._benchmark_run_dirs(case_dir):
            config_path = os.path.join(run_dir, "benchmark_config.yaml")
            if not os.path.exists(config_path):
                continue
            with open(config_path, encoding="utf-8") as handle:
                stored = yaml.safe_load(handle) or {}
            existing = dict(stored)
            existing.pop("execute_planned_trajectory", None)
            existing.pop("go_home_before_benchmark", None)
            if existing != config:
                raise RuntimeError("benchmark case lock differs; use a new benchmark_output_dir")
            if stored != config:
                self._write_yaml(config_path, config)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self._benchmark_slug(self.default_planner_id)}_seed{self.planner_random_seed}_{stamp}"
        run_dir = os.path.join(case_dir, stem)
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(case_dir, f"{stem}_{suffix}")
            suffix += 1
        os.makedirs(run_dir)
        self._write_yaml(os.path.join(run_dir, "benchmark_config.yaml"), config)
        return case_dir, run_dir, config

    @classmethod
    def _find_existing_goals(cls, case_dir, run_dir):
        for candidate_dir in cls._benchmark_run_dirs(case_dir):
            if os.path.abspath(candidate_dir) == os.path.abspath(run_dir):
                continue
            path = os.path.join(candidate_dir, "generated_goals.csv")
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _benchmark_slug(value):
        return "".join(char if char.isalnum() or char in "_.-" else "_" for char in str(value))

    @staticmethod
    def _write_results(path, rows):
        write_results(path, rows)

    @staticmethod
    def _write_benchmark_summary(path, rows, expected, run_mode, status="completed", reason=""):
        write_summary(path, rows, expected, run_mode, status, reason)

    def run_benchmark(self):
        case_dir = os.path.abspath(self.benchmark_output_dir) if self.benchmark_output_dir else None
        run_dir = None
        rows = []
        try:
            case_dir, run_dir, _config = self._prepare_benchmark_artifacts()
            self.setup_scene()
            if not self._wait_for_complete_joint_state(self.benchmark_startup_joint_state_timeout_s, "benchmark start"):
                raise RuntimeError("runtime_not_ready: missing complete joint state")
            if self._as_bool(self.get_parameter("auto_add_obstacle").value): self.add_default_obstacle()
            pre_home_ok, pre_home_error = self._ensure_home()
            if not pre_home_ok:
                raise RuntimeError(f"home_reset_failed: {pre_home_error}")
            start = self.scene_benchmark.get("start_pose")
            if start is None: raise RuntimeError(f"scene {self.scene_name} lacks benchmark.start_pose")
            start_xyz, _ = self._parse_pose_values(self._parse_float_list(start), self._parse_float_list(self.get_parameter("target_rpy_deg").value))
            target_rpy = tuple(self._parse_float_list(self.get_parameter("target_rpy_deg").value))
            goals_path = os.path.join(run_dir, "generated_goals.csv")
            source_goals_path = self._find_existing_goals(case_dir, run_dir)
            if source_goals_path:
                goals = self._read_generated_goals_csv(source_goals_path, start_xyz, target_rpy)
                self._write_generated_goals_csv(goals, goals_path, start_xyz)
            else:
                goals = self._generate_benchmark_goals(
                    self.benchmark_repetitions, start_xyz, target_rpy
                )
                self._write_generated_goals_csv(goals, goals_path, start_xyz)
            for index, (xyz, rpy) in enumerate(goals, 1):
                row = {"run_index": index, "run_mode": self.run_mode, "planner_id": self.default_planner_id, "planner_random_seed": self.planner_random_seed, "plan_success": "false", "success": "false", "failure_phase": "none", "error_code": "", "goal_pose": "/".join(f"{value:.4f}" for value in (*xyz, *rpy)), "core_planning_time_s": 0.0, "optimized_joint_path_length_rad": 0.0, "execution_success": "not_run", "return_home_success": "not_run"}
                if self.benchmark_executes_trajectory and index > 1 and not self._ensure_home()[0]: row.update(failure_phase="home_reset", error_code="home_reset_failed")
                else:
                    result = self._plan_pose_from_home(self.make_pose_from_xyzrpy(xyz, rpy)); row.update(plan_success=str(result["success"]).lower(), error_code=result["error_code"], core_planning_time_s=f"{result['core_planning_time_s']:.6f}")
                    trajectory = result["trajectory"]
                    if trajectory is None: row["failure_phase"] = "goal_plan"
                    else:
                        row.update(optimized_joint_path_length_rad=f"{self._joint_trajectory_path_length(trajectory):.6f}")
                        self._publish_display_trajectory(trajectory)
                        if self.benchmark_executes_trajectory:
                            ok, code = self._execute_joint_trajectory(trajectory); row.update(execution_success=str(ok).lower(), failure_phase="none" if ok else "goal_execute", error_code="" if ok else code)
                            if ok and not self.go_home(): row.update(success="false", return_home_success="false", failure_phase="return_home", error_code="return_home_failed")
                            elif ok: row.update(success="true", return_home_success="true")
                        else: row.update(success="true")
                rows.append(row); self._write_results(os.path.join(run_dir, "results.csv"), rows)
            self._write_benchmark_summary(os.path.join(run_dir, "summary.md"), rows, self.benchmark_repetitions, self.run_mode)
        except Exception as exc:
            if run_dir:
                self._write_results(os.path.join(run_dir, "results.csv"), rows)
                self._write_benchmark_summary(os.path.join(run_dir, "summary.md"), rows, self.benchmark_repetitions, self.run_mode, "aborted", str(exc))
            elif case_dir:
                os.makedirs(case_dir, exist_ok=True)
                with open(os.path.join(case_dir, "benchmark_aborted.md"), "w", encoding="utf-8") as handle:
                    handle.write(f"# Planning benchmark aborted\n\nreason: {exc}\n")
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

        # 可选：demo 前先回 HOME
        if self.go_home_before_demo:
            if not self.go_home():
                self.get_logger().error("回 HOME 失败，终止 demo")
                return
        else:
            self.get_logger().info("go_home_before_demo=false，启动后保持当前机械臂初始状态")

        self.setup_scene()
        # 按需添加障碍物
        if self._as_bool(self.get_parameter("auto_add_obstacle").value):
            self.add_default_obstacle()

        while rclpy.ok():
            # 读取用户输入（目标位姿或命令）
            action, data = self.read_pose_or_command(
                "输入终点 pose: x y z [rx ry rz]，或输入 ik/planner/go home/recover"
            )

            if action == "go_home":
                self.go_home()
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
        self.go_home()
        while rclpy.ok():
            action, data = self.read_pose_or_command(
                "输入 IK 目标 pose: x y z [rx ry rz]，或输入 ik/planner/go home"
            )
            if action == "go_home":
                self.go_home()
            elif action == "switch_ik":
                self.set_ik(data)
            elif action == "switch_planner":
                self.set_planner(*data)
            elif action == "recover":
                self.get_logger().warn("IK 对比模式不管理场景；请使用 home。")
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
