"""Robot profile loading for myrobot_simulation.

Profiles are the single source of truth for robot-specific package names,
controllers, xacro files, spawn pose, and optional capabilities such as a
gripper or simulated camera.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from myrobot_common.launch_utils.yaml_loader import load_yaml


@dataclass(frozen=True)
class RobotProfile:
    name: str
    robot_name: str
    moveit_config_package: str
    moveit_config_name: str
    description_package: str
    sim_xacro: str
    semantic_file: str
    default_kinematics_file: str
    kinematics_fairino_file: str
    kinematics_kdl_file: str
    planning_pipeline_file: str
    controllers_file: str
    initial_positions_file: str
    has_gripper: bool
    has_camera: bool
    spawn_name: str
    spawn_xyz: List[float]
    spawn_rpy: List[float]
    arm_controller: str
    hand_controller: str
    arm_joints: List[str]
    hand_joints: List[str]
    moveit_controllers_file: str = ""
    planning_pipelines: List[str] = field(default_factory=lambda: ["fairino", "ompl"])
    default_planning_pipeline: str = "fairino"
    group_name: str = "robot_arm"
    planning_frame: str = "base_link"
    ee_frame_name: str = "tool0"
    servo_parameters_file: str = "config/servo_parameters_sim.yaml"

    @property
    def controller_names(self) -> List[str]:
        names = [self.arm_controller]
        if self.has_gripper and self.hand_controller:
            names.append(self.hand_controller)
        return names

    @property
    def spawner_controller_names(self) -> List[str]:
        return ["joint_state_broadcaster"] + [name.lstrip("/") for name in self.controller_names]


def _as_list(data: Dict[str, Any], key: str, default: List[Any]) -> List[Any]:
    value = data.get(key, default)
    return list(value) if value is not None else list(default)


def load_robot_profile(profile_name: str) -> RobotProfile:
    """Load a robot profile by name from myrobot_simulation/config/robots."""
    data = load_yaml("myrobot_simulation", f"config/robots/{profile_name}.yaml")
    return RobotProfile(
        name=profile_name,
        robot_name=data["robot_name"],
        moveit_config_package=data["moveit_config_package"],
        moveit_config_name=data["moveit_config_name"],
        description_package=data["description_package"],
        sim_xacro=data["sim_xacro"],
        semantic_file=data["semantic_file"],
        default_kinematics_file=data.get("default_kinematics_file", "config/kinematics.yaml"),
        kinematics_fairino_file=data.get("kinematics_fairino_file", data.get("default_kinematics_file", "config/kinematics.yaml")),
        kinematics_kdl_file=data.get("kinematics_kdl_file", data.get("default_kinematics_file", "config/kinematics.yaml")),
        planning_pipeline_file=data.get("planning_pipeline_file", "config/fairino_planning.yaml"),
        controllers_file=data["controllers_file"],
        moveit_controllers_file=data.get("moveit_controllers_file", ""),
        initial_positions_file=data["initial_positions_file"],
        has_gripper=bool(data.get("has_gripper", False)),
        has_camera=bool(data.get("has_camera", False)),
        spawn_name=data.get("spawn_name", "robot_arm"),
        spawn_xyz=_as_list(data, "spawn_xyz", [0.0, 0.0, 0.0]),
        spawn_rpy=_as_list(data, "spawn_rpy", [0.0, 0.0, 0.0]),
        arm_controller=data["arm_controller"],
        hand_controller=data.get("hand_controller", ""),
        arm_joints=_as_list(data, "arm_joints", ["j1", "j2", "j3", "j4", "j5", "j6"]),
        hand_joints=_as_list(data, "hand_joints", []),
        planning_pipelines=_as_list(data, "planning_pipelines", ["fairino", "ompl"]),
        default_planning_pipeline=data.get("default_planning_pipeline", "fairino"),
        group_name=data.get("group_name", "robot_arm"),
        planning_frame=data.get("planning_frame", "base_link"),
        ee_frame_name=data.get("ee_frame_name", "tool0"),
        servo_parameters_file=data.get("servo_parameters_file", "config/servo_parameters_sim.yaml"),
    )
