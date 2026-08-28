# myrobot_simulation 架构总览（整合版）

## 1. 重构目标与范围

本文件整合以下文档核心内容：

- `dual_move_group_migration.md`
- `myrobot_simulation_dataflow.md`
- `myrobot_simulation_refactor.md`
- `launch_parameter_flow.md`
- `model_asset_ownership.md`
- `robot_profiles.md`

目标是保留关键设计决策与使用方法，同时去掉重复叙述，形成单一维护入口。

## 2. 目录与职责

当前关键结构：

```text
myrobot_simulation/
  launch/
    gazebo.launch.py
    visual_grasping_sim.launch.py
    graspnet_grasping_sim.launch.py
    motion_planning_demo_sim.launch.py
  launch_utils/
    robot_profiles.py
    controllers.py
    sim_stack.py
    moveit_stack.py
    perception_stack.py
    launch_parsing.py
  myrobot_common.launch_utils/
    yaml_loader.py
  config/robots/
    fairino_arm_gripper_onbase.yaml
    fairino3_v6.yaml
    dummy.yaml
    common/
    fairino_arm/
    fairino3_v6/
    dummy/
  worlds/models/
  rviz/
  docs/
```

职责边界：

- `launch/` 只做编排，不写业务细节。
- `launch_utils/` 承担可复用构建逻辑。
- `config/robots/*.yaml` 作为机器人 profile 单一来源。
- `config/robots/*/*.xacro` 管模型资产组织。
- `worlds/models/` 管 Gazebo 复用模型资产。
- `motion_planning_demo_sim.launch.py` 通过 `run_mode` 提供交互规划、闭环 benchmark 与纯算法 benchmark；benchmark 归档由节点直接生成结构化 CSV/Markdown。

## 3. 双 move_group 设计

系统固定双实例：

- `/move_group_fairino/move_group`
- `/move_group_kdl/move_group`

用途：

- `fairino`：连接 `/move_group_fairino`，使用该 move_group 加载的 IK 插件。
- `kdl`：连接 `/move_group_kdl`，使用该 move_group 加载的 IK 插件。

`planning_pipeline_id` 独立选择规划管线；例如 `kdl + fairino/tube_birrt*` 表示用 KDL 求 IK，再用 Fairino pipeline 规划。

客户端选择策略：

1. 如果显式传入 `move_group_namespace`，直接使用该命名空间。
2. 否则按 `planning_client` 映射：
   - `fairino -> /move_group_fairino`
   - `kdl -> /move_group_kdl`

## 4. 启动数据流

### 4.1 `gazebo.launch.py`

链路：

```text
robot_profile
  -> load_robot_profile()
  -> base_simulation_actions()
      -> build_moveit_config()
      -> move_group_nodes()
      -> controller_spawner_actions()
```

结果：

- 启动 Gazebo、robot spawn、clock bridge、robot_state_publisher
- 启动双 `move_group`
- 启动 controller spawner
- 可选启动 RViz

默认定位：纯仿真/纯规划入口，不含 YOLO/Servo 业务节点。

### 4.2 业务 Sim launch

业务 Sim 入口复用 `gazebo.launch.py` 的基础栈，并额外按需要启动：

- `camera_bridge_nodes()`
- `servo_node()`

默认定位：视觉与伺服联动场景入口。

### 4.3 `graspnet_grasping_sim.launch.py`

复用 `gazebo.launch.py` 的 Gazebo、MoveIt 和相机桥接基础栈，并额外启动：

- `trajectory_retime_server`
- `handeye_publisher.py`
- `graspnet_inference_node.py`
- `graspnetl_grasping_node.py`

默认定位：Gazebo 仿真下的 GraspNet RGB-D 视觉抓取闭环入口，当前只执行“生成抓取候选 -> 抓取 -> 抬起”，不包含放箱。详细启动和调参说明见：

```text
myrobot_simulation/docs/graspnet_grasping_sim说明文档.md
```

## 5. 参数与配置流

### 5.1 `robot_profile` 流

```text
launch 参数 robot_profile
  -> config/robots/<profile>.yaml
  -> RobotProfile
  -> MoveIt/Gazebo/controller/Servo 统一配置注入
```

核心字段（示例）：

- `moveit_config_package`
- `sim_xacro`
- `kinematics_fairino_file`
- `kinematics_kdl_file`
- `planning_pipeline_file`
- `controllers_file`
- `spawn_name/spawn_xyz/spawn_rpy`
- `has_gripper/has_camera`

### 5.2 Fairino 算法参数流

`move_group_fairino` 注入：

- `fairino_planning.yaml`
- `myrobot_planning_core/config/common_planning_params.yaml`（dict）
- `myrobot_planning_core/config/ik_params.yaml`（dict）

注意：必须先 `load_yaml()` 读成字典后注入，不能把这两个文件路径字符串直接作为 `parameters` 传入。

## 6. PlanningScene 与障碍物

关键原则：

- Gazebo 里看见的模型，不等价于 MoveIt 规划碰撞体。
- 规划器是否避障，取决于 `PlanningScene` 中是否有 `CollisionObject`。

MPC demo 静态障碍物流：

```text
obstacle_stack.yaml
  -> obstacle_simulator
  -> /planning_scene
  -> /move_group_fairino/move_group
  -> scene->getWorld()->getObjectIds()
```

为避免双命名空间导致场景消息丢失，`move_group_nodes()` 已统一 remap：

- `planning_scene -> /planning_scene`
- `collision_object -> /collision_object`
- `attached_collision_object -> /attached_collision_object`

若日志出现：

```text
Planning obstacles aggregated: obs_count=0
```

优先排查：

1. `/planning_scene` 是否有发布。
2. `move_group` 是否收到了 remap 后的话题。
3. 启动入口是否使用了当前 `launch_utils/moveit_stack.py`。

## 7. 相机模型与桥接的双开关

两层开关独立：

1. `enable_camera_model`：控制 xacro 是否生成相机与传感器插件。
2. `enable_camera_bridge`：控制是否把 Gazebo 图像桥接到 ROS。

推荐组合：

- 纯规划：`enable_camera_model=false`
- YOLO/Servo：`enable_camera_model=true` 且 `enable_camera_bridge=true`

## 8. 资产与路径规范

允许：

- `package://`
- `model://`
- `get_package_share_directory()`

禁止在生产 launch/xacro/world 中硬编码：

- `/home/robot/...`
- `file:///home/robot/...`

Gazebo 模型资产统一放 `worlds/models/`，world 通过 `model://` 引用。

## 9. 迁移与兼容策略

旧入口文件可保留短期兼容 wrapper，但新开发只走 profile + `launch_utils` 链路。

建议流程：

1. 新增机器人 -> 新建 `config/robots/<name>.yaml`
2. 新建专属 xacro -> `config/robots/<name>/...`
3. 不修改核心 launch 编排，只扩展 profile 与资产

## 10. 常用验收命令

```bash
ros2 node list | grep move_group
ros2 service list | grep plan_kinematic_path
ros2 param get /move_group_fairino/move_group fairino.max_iterations
ros2 topic list | grep planning_scene
```

## 11. 维护建议

1. 入口 launch 控制在“参数声明 + 编排”层，避免重新堆业务细节。
2. 任何机器人差异优先落在 profile 和 xacro，不在 launch 写分支硬编码。
3. 任何算法调参变更都要在 `myrobot_planning_core` 参数源统一维护。
4. 文档只维护本文件，其他细分文档不再作为主入口。

## 12. 跨包 Profile 共享契约

`myrobot_simulation` 与 `myrobot_mpc_avoidance` 统一复用 `myrobot_simulation/config/robots/*.yaml`，字段语义保持一致：

- `moveit_config_name` / `moveit_config_package`
- `group_name`
- `planning_frame`
- `ee_frame_name`
- `arm_controller`
- `arm_joints`

`myrobot_simulation/launch/mpc_avoidance_demo_sim.launch.py` 使用固定的 Fairino on-base 配置，并自动注入：

- `MoveItConfigsBuilder(...)`
- `mpc_avoidance_node` 的 `group_name/joint_names/controller_topic`
- `mpc_avoidance_node_sim.py` 的 `group_name/ee_link/base_frame/joint_names`

因此切换模型只需改 launch 参数，不需要修改 Python/C++ 源码：

```bash
ros2 launch myrobot_simulation mpc_avoidance_demo_sim.launch.py
```
