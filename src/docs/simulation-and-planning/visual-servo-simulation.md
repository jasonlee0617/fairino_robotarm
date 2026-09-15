# Gazebo 位置伺服纯跟踪数据流

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
