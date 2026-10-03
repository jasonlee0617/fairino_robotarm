# Gazebo 位置伺服纯跟踪数据流

[返回技术文档中心](../README.md)

该入口验证“全局移动到目标上方，再由速度伺服连续跟踪”的闭环。它不执行夹爪、下降抓取、抬升或放置。

入口命令：

```bash
ros2 launch myrobot_simulation visual_position_servo_sim.launch.py
```

## 启动链路

`visual_position_servo_sim.launch.py` 启动 Gazebo、双 MoveIt
move_group、MoveIt Servo、D435 bridge、RViz、YOLO Kalman、轨迹重定时服务、
cube 运动控制器与 `visual_position_servo`。

相机默认使用 `640x480@60`，YOLO Kalman 发布每类别一个已选最优三维目标：

- `/cube_position_3d`
- `/elongated_object_position_3d`
- `/box_position_3d`
- `/stone_position_3d`

cube 控制器仍订阅 `/cube_auto_start`，仅在 cube 被选为跟踪目标且全局移动完成后启动。

## 任务状态机

```text
IDLE -> SEARCHING -> MOVING_TO_TARGET_ABOVE -> SERVO_TRACK
  -> RETURNING_HOME -> COMPLETED -> IDLE -> SEARCHING
```

异常路径：

```text
SERVO_TRACK -> SERVO_HALT_RECOVERY -> SEARCHING
ANY -> ERROR -> IDLE
```

`SEARCHING` 按 `visual_position_servo_params.yaml` 的 `target_priority` 选择新鲜目标；
`preferred_target` 始终被提升为第一优先级。进入 `SERVO_TRACK` 后锁定当前类型，
只有该目标超时或丢失才停止 Servo 并回到 `SEARCHING` 重新选择，避免运动中跳目标。

## 控制闭环

1. `SEARCHING` 将目标 `PointStamped` 变换到 `base_link`，构造现有的
   `target_above_pose`：`x/y` 为目标位置、`z` 为目标加 `above_offset`。
2. `MOVING_TO_TARGET_ABOVE` 保持现有 MoveIt client、规划器、姿态、速度、加速度和约束不变。
3. `SERVO_TRACK` 以 XYZ 误差和现有控制器参数发布
   `/servo_node/delta_twist_cmds`。
4. 达到既有对齐和目标静止门槛后，发布零 Twist，停止 Servo，回 Home。
5. `COMPLETED` 清理缓存并进入 `IDLE`；下一次收到新鲜检测后立即重复以上流程。

纯跟踪不执行夹爪、下降抓取、抬升或放置；`box` 不需要轴向消息，也可直接跟踪。

## 控制器与参数

`visual_servo_bringup/config/visual_position_servo_params.yaml` 是位置伺服参数源。运行时通过 `controller_type` 选择 `PID`、`PD`、`PI_FF`、`ADAPTIVE_PID`、`LADRC`、`NLADRC` 或 `MPC`；各控制器共享误差输入、速度输出和限幅执行链，不改变状态机职责。

## 安全与验证

- Servo 输出仍受速度、加速度、工作空间和消息新鲜度约束；
- 目标丢失或时间戳过期会先发布零 Twist，再退出跟踪；
- 软件停止不是安全等级急停，真实机械臂必须使用独立硬件安全链路。

```bash
ros2 launch myrobot_simulation visual_position_servo_sim.launch.py --show-args
ros2 topic echo /task_state
ros2 topic hz /servo_node/delta_twist_cmds
```

`--show-args` 只验证 Launch 接口；完整验收还需要检查 Gazebo 中的目标运动、跟踪误差、限幅和停止恢复。
