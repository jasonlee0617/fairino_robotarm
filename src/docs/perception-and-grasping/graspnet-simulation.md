# graspnet_grasping_sim.launch.py 说明文档

本文档说明 `myrobot_simulation/launch/graspnet_grasping_sim.launch.py` 的用途、启动前准备、启动命令、运行链路、验收方法和常见问题。该 launch 面向 Sim 下的 Fairino Arm 机械臂 GraspNet 视觉抓取，目标是完成最小闭环：

```text
Gazebo RGB-D 相机
  -> 支撑平面过滤后由 GraspNet 生成抓取候选
  -> translation + approach_axis * (depth + grasp_offset)
  -> MoveIt 规划并执行 target_grasp
  -> 夹爪闭合
  -> 抬起
```

当前流程只做到“抓取并抬起”，不包含放箱或二次视觉搜索。

## 1. 启动入口

工作区根目录：

```bash
cd $HOME/my-workspace/fairino_robotarm
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch myrobot_simulation graspnet_grasping_sim.launch.py
```

该 launch 内部会启动以下组件：

| 组件 | 作用 |
| --- | --- |
| `myrobot_simulation/launch/gazebo.launch.py` | 启动 Gazebo、Fairino Arm、MoveIt、RViz、相机模型和相机 bridge。 |
| `myrobot_common_ws/trajectory_retime_server/launch/retime_server.launch.py` | 提供轨迹 retime 服务。 |
| `hand_eye_calibration/handeye_publisher.py` | 发布手眼标定 TF，默认读取 `robot_calibration`。 |
| `graspnet_bringup.graspnet_inference_node` | 在 `graspnet` conda 环境中运行 GraspNet 推理，发布抓取候选。 |
| `graspnet_bringup/graspnetl_grasping_node.py` | 调用 GraspNet 推理服务，并用 MoveIt 执行抓取和抬起。 |

## 2. 启动前准备

### 2.1 ROS 工作区构建

先确认相关包已构建并安装：

```bash
cd $HOME/my-workspace/fairino_robotarm
source /opt/ros/humble/setup.bash
colcon build --packages-select graspnet_bringup myrobot_simulation trajectory_retime_server hand_eye_calibration visual_grasping_bringup
source install/setup.bash
```

检查可执行入口：

```bash
ros2 pkg executables graspnet_bringup
```

至少应能看到：

```text
graspnet_bringup graspnet_inference
graspnet_bringup graspnet_visual_grasping
```

### 2.2 GraspNet 环境

推理节点由 launch 自动执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate graspnet
source /opt/ros/humble/setup.bash
source $HOME/my-workspace/fairino_robotarm/install/setup.bash
python -m graspnet_bringup.graspnet_inference_node
```

因此启动前需要确认：

| 项目 | 当前 launch 使用值 |
| --- | --- |
| conda 初始化脚本 | `~/miniconda3/etc/profile.d/conda.sh` |
| conda 环境名 | `graspnet` |
| GraspNet baseline | `$HOME/manipulator_grasp/graspnet-baseline` |
| checkpoint | `$HOME/manipulator_grasp/logs/log_rs/checkpoint-rs.tar` |

快速检查：

```bash
test -f ~/miniconda3/etc/profile.d/conda.sh
test -d $HOME/manipulator_grasp/graspnet-baseline
test -f $HOME/manipulator_grasp/logs/log_rs/checkpoint-rs.tar
```

如果这些路径不存在，需要先修正 `myrobot_simulation/launch/graspnet_grasping_sim.launch.py` 中的 `conda_setup`、`baseline_dir` 或 `checkpoint_path`。

### 2.3 手眼标定和 TF

执行节点默认要求存在：

```text
base_link -> camera_color_optical_frame
```

该 TF 由机器人模型和 `handeye_publisher.py` 共同提供。启动后可检查：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

若 `graspnet_visual_grasping` 的状态长期停在 `waiting_tf`，优先检查手眼标定是否已发布，以及 `calibration_name` 是否仍为 `robot_calibration`。

## 3. 当前固定参数与优先级

`graspnet_grasping_sim.launch.py` 固定使用包内执行节点 YAML。`graspnet_visual_grasping` 执行节点参数优先级为：

```text
CLI ROS 参数 > graspnet_grasping_params.yaml > launch runtime dict > 节点代码默认值
```

launch runtime dict 只保留运行时动态值，例如 `use_sim_time` 和 SRDF `pos1` 解析结果；可调行为参数放在 `graspnet_grasping_params.yaml`。

默认路径来自 `get_package_share_directory("graspnet_bringup")`，也就是 install/share 下的 YAML，不会直接读取 src。修改 src YAML 后重新构建或使用 `--symlink-install` 生效：

```bash
# 方式 1：重新构建或使用 --symlink-install 后读取 install/share
cd $HOME/my-workspace/fairino_robotarm
source /opt/ros/humble/setup.bash
colcon build --packages-select graspnet_bringup myrobot_simulation --symlink-install
source install/setup.bash
```

### 3.1 Gazebo 与相机

| 参数 | 当前值 |
| --- | --- |
| `robot_profile` | `fairino_arm_gripper_inhand` |
| `world` | `visual_grasping` |
| `enable_rviz` | `true` |
| `use_sim_time` | `true` |
| `enable_servo` | `false` |
| `camera_fps` | `30` |
| `camera_image_width` | `640` |
| `camera_image_height` | `480` |
| `spawn_x/y/z` | `0.0 / 0.0 / 1.02` |

### 3.2 GraspNet 推理节点

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `rgb_topic` | `/camera/camera/color/image_raw` | RGB 图像输入。 |
| `depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` | 对齐到彩色图的深度输入。 |
| `camera_info_topic` | `/camera/camera/aligned_depth_to_color/camera_info` | 相机内参。 |
| `num_point` | `20000` | 点云采样点数。 |
| `top_k_publish` | `5` | 发布前 5 个候选。 |
| `min_valid_points` | `2000` | 支撑平面过滤后的有效点下限。 |
| `support_plane_distance_m` | `0.005` | 从支撑平面保留物体点的最小距离。 |
| `confirm_before_publish` | `true` | 推理完成后先弹出 Open3D 确认窗口，按 `E` 才发布抓取结果。 |
| `confirm_visual_top_k` | `50` | 确认窗口中显示的候选抓取数量。 |
| `confirm_window_name` | `GraspNet candidates: E=execute, B=best, ESC/Q=cancel` | 确认窗口标题和按键提示。 |

输出：

```text
/grasp/compute  std_srvs/srv/Trigger
/grasp/poses    geometry_msgs/msg/PoseArray
/grasp/scores   std_msgs/msg/Float32MultiArray
/graspnet_bringup/preview_best_pose   geometry_msgs/msg/PoseStamped
/graspnet_bringup/preview_best_score  std_msgs/msg/Float32
```

### 3.3 抓取执行节点

| 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `base_frame` | `base_link` | MoveIt 和目标 pose 基准坐标系。 |
| `camera_frame` | `camera_color_optical_frame` | GraspNet 候选输入坐标系。 |
| `ee_frame` | `tool0` | MoveIt 末端坐标系。 |
| `grasp_offset_m` | `0.0` | 沿 GraspNet approach axis 叠加到预测 depth 的偏移。 |
| `lift_distance` | `0.08` | 抓取后沿 `base_link` 的 `+Z` 抬起距离。 |
| `max_grasp_candidates` | `5` | 最多尝试的候选数量。 |
| `graspnet_to_ee_rpy_deg` | `[0.0, 0.0, 0.0]` | GraspNet 姿态到夹爪末端姿态的修正。 |
| `pregrasp_pose` | `(0.180, 0.25, 0.25, 0, -180, 0)` | 启动、可恢复失败以及成功 lift 后的唯一回位姿态。 |
| `debug_compare_target_pose` | `true` | 打印固定 Gazebo cube 的 world/base 坐标对比。 |
| `debug_target_world_xyz` | `[0.2, 0.35, 1.05]` | `visual_world.sdf` 中 cube 的 world 坐标。 |
| `debug_robot_spawn_xyz` | `[0.0, 0.0, 1.02]` | 当前 launch 中机器人 spawn 的 world 坐标。 |
| `enable_target_gate` | `true` | 只执行接近固定 cube 的 GraspNet 候选。 |
| `max_target_xy_error_m` | `0.12` | 目标门控允许的 XY 平面误差。 |
| `max_target_z_error_m` | `0.15` | 目标门控允许的 Z 方向误差。 |

执行状态发布到：

```text
/graspnet_bringup/state  std_msgs/msg/String
/robot/target_pose        geometry_msgs/msg/PoseStamped
/graspnet_bringup/selected_grasp_6d  std_msgs/msg/String
/graspnet_bringup/grasp_plan_6d      std_msgs/msg/String
```

常见状态：

```text
waiting_tf
waiting_graspnet
waiting_moveit
PREGRASP_POSE
open_gripper
compute_grasps
confirm_grasp
select_grasp
move_to_grasp
close_gripper
lift
return_pregrasp
done
lifted
pregrasp_pose_failed
compute_failed
no_grasp_result
no_executable_grasp
```

## 4. 运行验收

启动后另开终端：

```bash
cd $HOME/my-workspace/fairino_robotarm
source /opt/ros/humble/setup.bash
source install/setup.bash
```

检查节点：

```bash
ros2 node list | grep -E 'graspnet|move_group|handeye'
```

检查相机输入：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/aligned_depth_to_color/camera_info --once
```

检查 GraspNet 服务和输出：

```bash
ros2 service list | grep /grasp/compute
ros2 service call /grasp/compute std_srvs/srv/Trigger {}
ros2 topic echo /grasp/poses --once
ros2 topic echo /grasp/scores --once
```

检查抓取执行状态：

```bash
ros2 topic echo /graspnet_bringup/state
ros2 topic echo /graspnet_bringup/selected_grasp_6d
ros2 topic echo /graspnet_bringup/grasp_plan_6d
```

如果流程正常，状态会经过：

```text
waiting_tf / waiting_graspnet / waiting_moveit
home
open_gripper
compute_grasps
confirm_grasp
select_grasp
move_to_grasp
close_gripper
lift
lifted
```

其中等待状态是否出现取决于各组件启动速度。进入 `confirm_grasp` 后会弹出 Open3D 窗口，按 `S` 只显示最佳抓取姿态，按 `Space` 继续执行抓取；按 `Esc`、`Q` 或关闭窗口会取消本次抓取。

执行节点会在终端和 `/graspnet_bringup/selected_grasp_6d` 输出实际尝试的 6D 抓取姿态。该姿态已经从 `camera_color_optical_frame` 转换到 `base_link`，并已应用 `graspnet_to_ee_rpy_deg` 修正。

按 `S` 筛选最佳抓取姿态时，推理节点会发布 `/graspnet_bringup/preview_best_pose` 和 `/graspnet_bringup/preview_best_score`。执行节点收到后会转换到 `base_link`，并在终端和 `/graspnet_bringup/grasp_plan_6d` 汇总输出预览目标。按 `S` 不会发布 `/grasp/poses`，也不会触发机械臂执行。

```text
Preview Grasp plan score=0.1322 frame=base_link
  target_grasp xyz=(...) rpy_deg=(...) quat_xyzw=(...)
  target_lift  xyz=(...) rpy_deg=(...) quat_xyzw=(...)
```

`target_grasp` 是 GraspNet 输出转换到 `base_link` 后，沿其 approach axis 前移 `depth + grasp_offset_m` 的最终抓取位姿；`target_lift` 是抓取后沿 `base_link` 的 `+Z` 抬起位姿。

## 5. 数据流

```text
gazebo_yolo.launch.py
  -> camera_bridge_nodes()
  -> /camera/camera/color/image_raw
  -> /camera/camera/aligned_depth_to_color/image_raw
  -> /camera/camera/aligned_depth_to_color/camera_info

graspnet_inference_node.py
  -> 同步 RGB + Depth + CameraInfo
  -> 支撑平面过滤和点云采样
  -> GraspNet checkpoint 推理
  -> 按 S 时发布 /graspnet_bringup/preview_best_pose + /graspnet_bringup/preview_best_score
  -> /grasp/poses + /grasp/scores

graspnetl_grasping_node.py
  -> 调用 /grasp/compute
  -> 按 S 时预览 best pose，并转换到 base_link 输出 6D 汇总
  -> 将 camera_color_optical_frame 下的候选转换到 base_link
  -> 按 score 排序尝试候选
  -> MoveIt 全局规划到 target_grasp
  -> close gripper
  -> MoveIt 全局规划到 target_lift
```

## 6. 调参入口

### 6.1 GraspNet 没有候选或有效点太少

优先调整：

```python
"-p min_valid_points:=2000 "
"-p depth_min_m:=0.05 "
"-p depth_max_m:=5.0 "
```

如果日志出现：

```text
Too few valid points after support-plane filtering
```

说明支撑平面之外的物体点不足，或深度输入为空。先检查相机图像、深度和 `base_link <- camera` 的 TF，再调整平面距离阈值。

### 6.2 抓取姿态方向不对

优先调整：

```python
"graspnet_to_ee_rpy_deg": [0.0, 0.0, 0.0],
```

当前直接规划到 `target_grasp`，抓取高度来自 GraspNet 输出并经过 TF 转换后的位姿。如果夹爪接近方向与仿真中显示的不一致，先修正 `graspnet_to_ee_rpy_deg`。

### 6.3 固定 cube 目标门控

当前固定测试场景中，cube 在 `visual_world.sdf` 的 world 坐标是：

```xml
<pose>0.2 0.35 1.05 0 0 0</pose>
```

但执行节点打印的是 `base_link` 坐标。当前 launch 显式使用机器人 spawn：

```text
spawn_xyz=(0.0, 0.0, 1.02)
```

所以 cube 中心在 `base_link` 下的期望位置约为：

```text
expected_base_xyz=(0.2, 0.35, 0.03)
```

如果终端中出现类似 `xyz=(0.1945,0.3480,0.0297)`，说明该候选位置与 cube 实际上是对齐的。若出现明显偏离该位置的候选，`enable_target_gate=true` 会按 `max_target_xy_error_m` 和 `max_target_z_error_m` 自动跳过，避免机械臂抓向非 cube 区域。

### 6.4 MoveIt 规划失败

优先检查：

```bash
ros2 service list | grep plan_kinematic_path
ros2 action list | grep follow_joint_trajectory
ros2 control list_controllers
```

执行节点默认依赖：

```text
/move_group_fairino
/move_group_kdl
/robot_arm_controller/follow_joint_trajectory
/hand_controller/follow_joint_trajectory
```

如果状态停在 `waiting_moveit`，通常是 move_group、controller 或 action server 尚未 ready。

## 7. 常见问题

### 7.1 `conda: command not found`

检查：

```bash
test -f ~/miniconda3/etc/profile.d/conda.sh
```

如果 conda 安装路径不同，需要修改 launch 中的 `conda_setup`。

### 7.2 `No synchronized RGB/Depth/CameraInfo received yet`

说明 GraspNet 推理节点还没有收到同步的三路相机消息。检查：

```bash
ros2 topic list | grep camera
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/aligned_depth_to_color/camera_info --once
```

### 7.3 TF 位姿变换失败（`TF pose transform failed`）

检查：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

如果没有 TF，先排查机器人 profile 是否为 `fairino_arm_gripper_inhand`，以及 `handeye_publisher.py` 是否正常启动。

### 7.4 SDF 位置和终端 3D xyz 差很多

先确认比较的是同一个坐标系。SDF 中的 `<pose>` 是 Gazebo world 坐标，`selected_grasp_6d` 和 `grasp_plan_6d` 输出的是 `base_link` 坐标。当前场景要先减去机器人 spawn 高度 `1.02m`，所以 `world z=1.05` 对应 `base_link z≈0.03`。

### 7.5 Open3D 窗口阻塞

当前 launch 设置：

`confirm_before_publish=true` 会在 `/grasp/compute` 后弹出人工确认窗口。该窗口会阻塞推理服务，直到按 `Space` 确认、按 `Esc/Q` 取消或关闭窗口。

## 8. 静态检查命令

文档或 launch 修改后建议执行：

```bash
cd $HOME/my-workspace/fairino_robotarm/src
python3 -m py_compile \
  myrobot_simulation/launch/graspnet_grasping_sim.launch.py \
  graspnet_bringup/graspnet_bringup/graspnet_inference_node.py \
  graspnet_bringup/graspnet_bringup/graspnetl_grasping_node.py
git diff --check
```

构建检查：

```bash
cd $HOME/my-workspace/fairino_robotarm
source /opt/ros/humble/setup.bash
colcon build --packages-select graspnet_bringup myrobot_simulation
```
