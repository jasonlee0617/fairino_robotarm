# motion_planning_demo_sim.launch.py 路径规划与 IK 对比 Demo 说明

本文档说明唯一规划入口 `myrobot_simulation/launch/motion_planning_demo_sim.launch.py`。它以同一节点复用场景、MoveIt 与 IK 配置：`run_mode:=interactive` 提供终端规划/IK 对比；`run_mode:=goal_collection` 采集可复用目标集；`run_mode:=benchmark_execution` 运行完整 start -> goal -> start 闭环 benchmark；`run_mode:=benchmark_algorithm` 先到配置起点一次后仅统计规划算法结果，并在节点退出后关闭整套 launch。

旧 benchmark 入口、节点与诊断脚本已删除，不保留兼容包装。

## Benchmark 使用与归档

先采集目标集，再运行 benchmark：

```bash
ros2 launch myrobot_simulation motion_planning_demo_sim.launch.py run_mode:=goal_collection
ros2 launch myrobot_simulation motion_planning_demo_sim.launch.py run_mode:=benchmark_algorithm
```

`goal_collection_dir` 与 `benchmark_output_dir` 分别归档目标集和运行结果；`planner_random_seed` 是实验变量。目标集身份包含场景、goal seed、数量、手动 `start_id`、完整 `start_joints`、IK 模式、固定姿态、布局签名和采样约束，因此 SR/MR、算法和 planner seed 均加载同一批末端位姿。ROS 运行时仅接受 `format_version: 4` 的目标集；旧 V1/V2/V3 归档不能被 ROS 采集或 benchmark 复用。若同名 `start_id` 的完整身份不一致，直接报错并要求更换 `start_id`。

```text
benchmark_goal_collection/<scene>/goal_seed07_n30_start_id_home/
  goal_set.csv                 # 仅目标索引与末端位姿
  goal_set_manifest.yaml       # format_version=4、collection_key、采样身份、CSV SHA256
  goal_sampling_stats.yaml     # 接受率和各类拒绝统计

trajectory_plan_benchmark_sample/<scene>/goal_seed07_n30/<mr|sr>/<planner_slug>/planner_seed07/<timestamp>/
  run_manifest.yaml
  results.csv
  anytime_trace.csv
  root_diagnostics.csv
  algorithm_diagnostics.csv
  trajectory_paths.csv
```

Benchmark 允许 `rrt*`、`informed_rrt*`、`birrt*`、`aapf_birrt*`、`mire_biait*` 和 `prm`；标准 `rrt` 可单独调用但不参与归档对比。Benchmark 禁用公共 PathOptimizer，故 `results.csv` 使用原始的 `joint_path_length_rad` 和 `tcp_path_length_m` 字段；通用汇总只报告成功率、时间、路径长度及状态/运动碰撞检查与无效边均值。

## 1. 总体作用

`motion_planning_demo_sim.launch.py` 是交互式路径规划、IK 对比和 benchmark 的顶层入口。它完成三件事：

1. 启动 Gazebo、robot_state_publisher、ros2_control、MoveIt move_group、RViz 等仿真与规划基础设施。
2. 加载路径规划场景配置，并可同步发布到 MoveIt PlanningScene、RViz Marker 和 Ignition Gazebo YAML 动态模型。
3. 启动 `motion_planning_node_sim.py`，由用户在终端选择路径规划或 Fairino/KDL IK 对比。

典型启动：

```bash
ros2 launch myrobot_simulation motion_planning_demo_sim.launch.py
```

launch 参数可覆盖 `_SCENE_DEFAULTS` 与 `_NODE_DEFAULTS`；终端内可切换 IK 客户端和规划器。

## 2. 启动数据流

整体启动链路如下：

```text
motion_planning_demo_sim.launch.py
  |
  |-- Include gazebo.launch.py
  |     |
  |     |-- launch_utils.sim_stack.base_simulation_actions()
  |     |     |
  |     |     |-- 生成 robot_description / robot_description_semantic
  |     |     |-- 启动 Gazebo / robot_state_publisher / ros2_control
  |     |     |-- 启动 move_group_fairino
  |     |     |-- 启动 move_group_kdl
  |     |     |-- 启动 fairino_cartesian_path_server
  |     |     |-- 启动 RViz
  |
  |-- TimerAction(5s)
        |
        |-- motion_planning_node_sim.py
              |
              |-- 读取 launch 参数
              |-- 终端选择路径规划或 IK 对比
              |-- 路径规划模式加载场景、发布障碍物并调用 MoveIt pipeline
              |-- IK 对比模式仅调用两个 /compute_ik，不加载或修改场景
```

`TimerAction(period=5.0)` 的作用是给 Gazebo、move_group 和控制器一点启动时间，避免 demo 节点过早请求规划服务。

## 3. 参数加载与优先级

参数来源优先级由高到低：

1. `motion_planning_demo_sim.launch.py` 内的 `_SCENE_DEFAULTS` / `_NODE_DEFAULTS`。
2. `gazebo.launch.py` 中的 fallback 默认值。
3. `motion_planning_node_sim.py` 的节点默认值。

`motion_planning_demo_sim.launch.py` 会把同一组场景参数同时传给：

- `gazebo.launch.py`：用于保持上层 launch 参数一致。
- `motion_planning_node_sim.py`：仅路径规划模式加载场景、发布障碍物并交互规划。

需要注意：场景仅由 `motion_planning_node_sim.py` 的路径规划模式通过 `SceneEnvironmentManager` 加载和发布；IK 对比模式不加载或修改场景。

## 4. 关键参数说明

### 4.1 机器人与仿真参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `robot_profile` | `fairino_arm_gripper_onbase` | 选择机器人配置 profile。影响 xacro、MoveIt config、控制器、规划管线列表。 |
| `world` | `empty` | Ignition Gazebo world 名称，同时传给 Gazebo obstacle spawn 的 `-world`。 |
| `use_sim_time` | `true` | 仿真时间开关。 |
| `enable_rviz` | `true` | 是否启动 RViz。 |
| `rviz_config` | `rviz/fairino_planning_test.rviz` | RViz 配置文件。 |

### 4.2 IK 和 MoveIt 接口参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `planning_client` | `fairino` | 选择默认规划客户端：`fairino` 或 `kdl`。本质上决定节点连接哪个 move_group 命名空间。 |
| `move_group_namespace` | 空 | 显式覆盖默认规划客户端的 move_group namespace，例如 `/move_group_kdl`。 |
| `group_name` | `robot_arm` | MoveIt planning group。 |
| `base_frame_name` | `base_link` | 输入 pose 与障碍物默认所在坐标系。 |
| `ee_frame_name` | `tool0` | 末端执行器 link。 |
| `joint_names` | `j1,j2,j3,j4,j5,j6` | arm joint 顺序。 |
| `start_joints` | `-1.1170,-1.6214,1.5465,-1.5877,-1.6368,0.0` | 按 `joint_names` 顺序的关节起点（弧度）；直接校验和执行，不经起点 IK。 |
| `start_id` | `home` | 仅允许字母和数字的手动起点标识；目标集目录使用它区分不同起点。 |

`planning_client=fairino` 时，节点默认使用 `/move_group_fairino`：

```text
motion_planning_node_sim.py -> pymoveit2 -> /move_group_fairino
```

`planning_client=kdl` 时，节点默认使用 `/move_group_kdl`：

```text
motion_planning_node_sim.py -> pymoveit2 -> /move_group_kdl
```

如果设置了 `move_group_namespace`，则该显式命名空间优先于 `planning_client` 推导。

### 4.3 规划管线与算法参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `default_pipeline_id` | `fairino` | MoveIt planning pipeline，例如 `fairino` 或 `ompl`。 |
| `default_planner_id` | `birrt*` | planner id，例如 `mire_biait*`、`birrt*`、`rrt`、`rrt*`、`informed_rrt*`、`prm`、`aapf_birrt*`、`RRTConnect`。 |
| `target_rpy_deg` | `0,-180,0` | 用户只输入 `x y z` 时使用的固定末端姿态，单位为度。 |
| `go_start_before_demo` | `false` | interactive 规划 demo 开始前是否先到配置起点。 |

推荐静态配置值：

- Fairino BiRRT*: `default_pipeline_id="fairino"`, `default_planner_id="birrt*"`
- Fairino RRT*: `default_pipeline_id="fairino"`, `default_planner_id="rrt*"`
- Fairino Informed-RRT*: `default_pipeline_id="fairino"`, `default_planner_id="informed_rrt*"`
- Fairino PRM: `default_pipeline_id="fairino"`, `default_planner_id="prm"`
- Fairino AAPF-BiRRT*: `default_pipeline_id="fairino"`, `default_planner_id="aapf_birrt*"`
- OMPL RRTConnect: `default_pipeline_id="ompl"`, `default_planner_id="RRTConnect"`

`default_pipeline_id` 和 `default_planner_id` 会进入 `motion_planning_node_sim.py`，然后设置到 `pymoveit2.MoveIt2`：

```python
self.moveit2_arm.pipeline_id = pipeline
self.moveit2_arm.planner_id = algorithm
```

之后 `move_to_pose(..., cartesian=False)` 会走 MoveIt 全局规划，而不是 Cartesian path。

### 4.4 场景参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `scene_assets_dir` | `myrobot_simulation/config/scenes` | 兼容保留：未显式设置 `scene_config_file` 时的场景 YAML 目录。 |
| `scene_config_file` | `myrobot_simulation/config/scenes/pathplanning_scenes_params.yaml` | 场景 YAML。 |
| `scene_name` | `single_obstacle` | 选择 YAML 中的场景 key。 |
| `spawn_sim_scene_models` | `true` | 是否依据场景 YAML 动态生成并 spawn Ignition Gazebo 障碍物。 |
| `publish_planning_scene` | `true` | 是否把 obstacle 发布到 MoveIt PlanningScene，规划避障以它为权威。 |
| `publish_obstacle_markers` | `true` | 是否发布 RViz MarkerArray。 |
| `obstacle_marker_topic` | `/demo_pathplanning/obstacle_markers` | RViz 障碍物 marker topic。 |

障碍物几何唯一来自 `scene_config_file` 指向的
`pathplanning_scenes_params.yaml`。配置文件不存在、为空、场景不存在或选定
场景没有障碍物时，节点直接报错，不再回退到程序内置的单障碍物。

## 5. 场景选择与发布流程

场景文件：

```text
myrobot_simulation/config/scenes/pathplanning_scenes_params.yaml
```

当前包含：

- `single_obstacle`
- `multi_obstacle_3d_avoidance`
- `dense_multi_obstacle_3d_avoidance`
- `dense_hard_multi_obstacle_3d_avoidance`：错位竖直障碍门与窄通道。
- `dense_extreme_multi_obstacle_3d_avoidance`：保留 dense 的 6 个基础障碍物，恢复相同的目标包围范围，再加入 3 个轻量旋转/错位障碍门组成多阶段通道。

后三个场景构成基线、困难和极限三级测试集。升级场景只新增 YAML 布局，不修改已有
`dense_multi_obstacle_3d_avoidance`，因此旧 benchmark 结果仍可复用。所有场景共用
launch 的 `start_joints`，目标仍由障碍物 AABB 内的确定性拒绝采样生成。
目标筛选只要求目标姿态有有效 IK、目标状态无碰撞并满足间距约束，不用某个规划器
预先筛掉“难规划”的目标。

难度报告应至少包含障碍物数量/形状、padding 后的最小通道间隙、目标接受率、目标
表面距离分布、起点到目标直线阻断率和每个目标的有效 IK 数量。目标接受率和直线
阻断率用于描述场景难度，不能反向用算法成功率定义场景难度。

Extreme 的障碍物设计按机械臂尺寸进行约束：Fairino 机械臂的主要连杆长度约为
`0.28 m` 和 `0.24 m`，末端工具偏移约为 `0.1168 m`。因此障碍门不能只按 TCP
点的间隙设计；门的位置和高度先保持在目标 AABB 内，再由带 `0.03 m` padding
的 MoveIt 完整机器人状态检查验证终点和路径。当前 Extreme 使用中心约为
`(0.36,0.10,0.26)`、`(0.44,0.00,0.26)`、`(0.58,-0.08,0.26)` 的小型错位门，
尺寸为 `0.04 x 0.08 x 0.30 m`，旋转角为 `15°/0°/-15°`。这代表多阶段绕障压力，
不把过大的障碍物直接放入目标采样边界。

目标生成结束后，运行目录中的 `goal_sampling_stats.yaml` 记录候选总数、几何拒绝、
`rejected_ik_geometry`、`rejected_ik_other`、终点状态碰撞、间距拒绝、接受率和已
接受目标的障碍物表面距离。IK 的 `D_domain` 拒绝与终点状态碰撞必须分开报告，不能
混为同一种失败。

每个 scene 可包含：

```yaml
benchmark:
  difficulty_level: hard
  difficulty_focus: narrow_staggered_corridors
obstacles:
  - name: xxx
    shape: box | cylinder | sphere
    pose: [x, y, z, rx, ry, rz]
    size: [sx, sy, sz]      # box
    radius: 0.05            # cylinder/sphere
    height: 0.30            # cylinder
    color: [r, g, b, a]
```

加载路径：

```text
motion_planning_node_sim.py（仅路径规划模式）
  |
  |-- SceneEnvironmentManager
        |
        |-- SceneLoader
        |     |-- 读取 pathplanning_scenes_params.yaml
        |     |-- 生成 SceneObstacle 列表
        |
        |-- PlanningSceneManager
        |     |-- 发布 CollisionObject 到 /planning_scene
        |
        |-- MarkerPublisher
        |     |-- 发布 MarkerArray 到 /demo_pathplanning/obstacle_markers
        |
        |-- GazeboSceneSpawner
              |-- ros2 run ros_gz_sim create -string <generated-sdf>
```

### 5.1 MoveIt PlanningScene

规划碰撞权威是 MoveIt PlanningScene。也就是说，即使 Gazebo 模型不可见，只要 `publish_planning_scene=true`，规划器仍应避开这些障碍物。

`PlanningSceneManager` 将 YAML 中的几何转换为 `shape_msgs/SolidPrimitive`：

- `box` -> `SolidPrimitive.BOX`
- `cylinder` -> `SolidPrimitive.CYLINDER`
- `sphere` -> `SolidPrimitive.SPHERE`

发布 topic：

```text
/planning_scene
```

`move_group_fairino` 和 `move_group_kdl` 在 launch 中都 remap 到根 `/planning_scene`，因此两个 move_group 能接收同一份障碍物。

### 5.2 RViz Marker

RViz marker 只用于可视化，不参与碰撞判断。发布 topic：

```text
/demo_pathplanning/obstacle_markers
```

若 RViz 中看不到障碍物，先确认：

- `publish_obstacle_markers:=true`
- RViz 添加了 `MarkerArray`
- topic 指向 `/demo_pathplanning/obstacle_markers`

### 5.3 Ignition Gazebo 静态模型

`GazeboSceneSpawner` 直接从 YAML 的 `shape`、`size` / `radius` / `height`、`color` 和 `pose` 生成静态 SDF：

```bash
ros2 run ros_gz_sim create \
  -world <world> \
  -string <generated-sdf> \
  -name <scene_name>_<model_name> \
  -x <x> -y <y> -z <z> \
  -R <roll> -P <pitch> -Y <yaw> \
  -allow_renaming true
```

Gazebo 的 visual 和 collision 均使用 YAML 原始尺寸。路径规划避障仍以 MoveIt PlanningScene 为准，后者会额外应用 `planning_scene_obstacle_padding_m` 安全边界。

## 6. IK 求解器与 move_group 关系

`gazebo.launch.py` 最终通过 `launch_utils.moveit_stack.move_group_nodes()` 启动两个 move_group：

```text
/move_group_fairino/move_group
/move_group_kdl/move_group
```

两者差异：

- `move_group_fairino` 加载 Fairino 自定义 IK，可使用 `fairino` 或 `ompl` planning pipeline。
- `move_group_kdl` 加载 KDL kinematics；在当前 profile 下也可加载 Fairino planning pipeline 和 RRT*/BiRRT*/AAPF/Tube 参数。

`motion_planning_demo_sim.launch.py` 的 `planning_client` 决定节点默认连接哪个 move_group：

```text
planning_client=fairino -> /move_group_fairino
planning_client=kdl     -> /move_group_kdl
```

如果设置 `move_group_namespace="/move_group_xxx"`，则显式 namespace 覆盖自动选择。

IK 和规划管线是独立选择的：`planning_client` 只决定 MoveIt/IK client，`default_pipeline_id` 和 `default_planner_id` 共同决定轨迹规划管线与算法。因此允许 `planning_client="kdl", default_pipeline_id="fairino", default_planner_id="birrt*"`，也允许 `planning_client="fairino", default_pipeline_id="ompl", default_planner_id="RRTConnect"`。

## 7. 轨迹规划算法选择

规划算法分两层：

1. `planning_pipeline`：MoveIt pipeline 名称。
2. `planning_algorithm`：该 pipeline 内的 planner id。

Fairino pipeline 示例：

`default_pipeline_id="fairino"`，`default_planner_id` 可设为 `mire_biait*`、`birrt*`、`rrt`、`rrt*`、`informed_rrt*`、`prm`、`aapf_birrt*`。标准 `rrt` 不允许进入 benchmark。

OMPL 示例：

`default_pipeline_id="ompl"`，`default_planner_id="RRTConnect"`。

Fairino 相关参数由 `moveit_stack.py` 注入到启用 Fairino pipeline 的 move_group；当前 `fairino_arm_gripper_onbase`、`fairino_arm_gripper_inhand` 和 `fairino_arm_gripper_calibration_onbase` profile 会同时给 `/move_group_fairino` 和 `/move_group_kdl` 注入：

```text
myrobot_planning_core/config/common_planning_params.yaml
myrobot_planning_core/config/aapf_birrt_star_params.yaml
myrobot_planning_core/config/birrt_star_params.yaml
myrobot_planning_core/config/rrt_star_params.yaml
myrobot_planning_core/config/ik_params.yaml
```

因此，规划算法的具体采样、步长、优化、IK 评分等底层参数不在 `motion_planning_demo_sim.launch.py` 中定义，而由 Fairino planning core 的 YAML 管理。

## 8. 交互式规划流程

`motion_planning_node_sim.py` 启动后先显示模式菜单：`1` 为碰撞感知路径规划，`2` 为 Fairino/KDL 原始 IK 对比，`q` 退出。

路径规划模式的主循环为：

1. 设置规划客户端和规划器：
   ```text
   set_ik(planning_client)
   set_planner(default_pipeline_id, default_planner_id)
   ```
2. 若 `auto_add_obstacle=true`，发布当前场景障碍物。
3. 若 `go_start_before_demo=true`，直接校验并执行 `start_joints`。
4. 循环读取起点 pose：
   ```text
   x y z rx ry rz
   ```
   或只输入：
   ```text
   x y z
   ```
   此时姿态使用 `target_rpy_deg`。
5. 若 `move_to_start=true`，先规划执行到起点；否则当前机械臂状态就是真实规划起点。
6. 读取终点 pose。
7. 调用：
   ```text
   move_to_pose(goal_pose, cartesian=False)
   ```
   即全局规划避障。
8. 用户选择继续或退出。

支持控制命令：

```text
go start
recover
```

`recover` 会清理场景、回配置起点、重置规划器并清空末端轨迹。起点无 IK 或起点状态碰撞时直接失败，不回退到关节常量。

IK 对比模式对同一目标调用 `/move_group_fairino/compute_ik` 和 `/move_group_kdl/compute_ik`，报告成功状态、错误码、耗时和关节解差异；它不会加载或修改规划场景，也不会对选中的 IK 解做碰撞规划。

## 9. 相关代码文件关系

```text
myrobot_simulation/launch/motion_planning_demo_sim.launch.py
  顶层路径规划与 IK 对比 launch，声明参数，include Gazebo stack，启动合并节点。

myrobot_simulation/launch/gazebo.launch.py
  通用 Gazebo/MoveIt 启动入口，接收 planning_demo 透传参数。

myrobot_simulation/launch_utils/moveit_stack.py
  构建 MoveIt config，启动 move_group_fairino、move_group_kdl、fairino_cartesian_path_server。

myrobot_simulation/scripts/motion_planning_node_sim.py
  终端模式菜单；路径规划模式负责场景和碰撞规划，IK 模式负责 Fairino/KDL 原始 IK 对比与直接关节执行。

myrobot_simulation/scripts/pathplanning_scene_tools.py
  场景工具模块：YAML 解析、PlanningScene 发布、RViz marker、Gazebo SDF 动态生成与 spawn。

myrobot_simulation/config/scenes/pathplanning_scenes_params.yaml
  路径规划 benchmark 场景定义，也是 Gazebo、RViz 和 MoveIt 共用的障碍物几何来源；MoveIt 额外施加规划安全边界。
```

## 10. 推荐测试命令

基础单障碍物场景：

设置 `scene_name="single_obstacle"`；该场景也必须存在于
`pathplanning_scenes_params.yaml` 中，`default_planner_id` 可设为 `birrt*`。

论文简易三维避障场景：

当前静态默认值即 `multi_obstacle_3d_avoidance + aapf_birrt* + spawn_sim_scene_models=true`。

论文高密度三维避障场景：

设置 `scene_name="dense_multi_obstacle_3d_avoidance"`。

困难/极限分层场景：

分别设置 `scene_name="dense_hard_multi_obstacle_3d_avoidance"` 或
`scene_name="dense_extreme_multi_obstacle_3d_avoidance"`。建议对每个场景使用相同
的 goal seed 和 planner seed，并在同一场景目标文件上配对比较所有规划器。

KDL + OMPL 对照：

设置 `planning_client="kdl"`、`default_pipeline_id="ompl"`、`default_planner_id="RRTConnect"`。

## 11. 常见问题排查

### RViz 有障碍物，Gazebo 没有

检查：

```bash
NODE_PARAMS["spawn_sim_scene_models"] = True
```

YAML 中每个 obstacle 只需包含对应形状的几何参数与 `pose`。

### Gazebo 有障碍物，但规划穿过去

说明 Gazebo 障碍物已生成，但 MoveIt PlanningScene 可能未发布或 move_group 未收到。检查：

```bash
publish_planning_scene:=true
```

并查看 `/planning_scene` 是否有 collision object。

### 更换算法无效

确认同时设置：

`default_pipeline_id="fairino"`、`default_planner_id="birrt*"`

或：

`default_pipeline_id="ompl"`、`default_planner_id="RRTConnect"`

`planning_algorithm` 必须是对应 pipeline 中存在的 planner id。

### 输入 6D pose 后 IK 失败

主评测建议固定姿态，只改变 XYZ。若输入任意 RPY，测试结果会混入 IK 可达性、腕部奇异、姿态约束难度，不再是纯避障算法对比。

推荐主评测输入：

```text
x y z
```

并通过 launch 固定：

```bash
target_rpy_deg:=0,-180,0
```
