# Fairino Robot Arm

这是一个面向 Fairino 六轴机械臂的 ROS 2 综合工作区，覆盖运动规划、Gazebo 仿真、RGB-D 感知、YOLO/GraspNet 抓取、视觉伺服、实时语音 LLM 控制、手眼标定和 MPC 动态避障。

本文用于回答三个问题：仓库能做什么、各目录负责什么、第一次进入仓库应从哪里开始。实现原理、参数和故障排查请进入[技术文档中心](src/docs/README.md)。

> [!IMPORTANT]
> 本仓库同时包含仿真、云端模型和真实机械臂相关代码。静态检查或单元测试通过，不等于 Gazebo、Qwen 云连接或真实机械臂已经完成验收。实机运行前必须独立检查急停、限位、速度、碰撞环境和工具安装。

## 能力概览

| 能力 | 当前实现 | 从这里开始 |
| --- | --- | --- |
| 运动规划与 IK | MoveIt 2、自定义 RRT 系列、Fairino/KDL IK、轨迹重定时 | [规划算法](src/docs/simulation-and-planning/规划算法结构说明.md) |
| Gazebo 仿真 | 机器人、相机、控制器、规划场景和业务 Launch 统一编排 | [仿真架构](src/docs/simulation-and-planning/仿真环境架构说明.md) |
| YOLO 视觉抓取 | RGB-D、OBB、TF、目标选择、抓放状态机 | [YOLO 视觉抓取](src/docs/perception-and-grasping/yolov8-visual-grasping.md) |
| GraspNet 抓取 | 点云抓取候选、姿态选择、MoveIt 执行 | [GraspNet 仿真](src/docs/perception-and-grasping/graspnet-simulation.md) |
| 视觉伺服 | PID、MPC、LADRC、NLADRC 位置伺服与状态机 | [位置伺服](src/docs/simulation-and-planning/visual-servo-simulation.md) |
| LLM 任务控制 | 本地 KWS、Qwen Realtime、Function Calling、YOLO 场景绑定、安全 Preview 与 MoveIt 执行 | [LLM 控制](src/docs/perception-and-grasping/llm-yolo-control.md) |
| 手眼标定 | 自动/半自动 Eye-in-Hand、半自动 Eye-on-Base | [手眼标定](src/docs/手眼标定/手眼标定文档说明.md) |
| MPC 动态避障 | acados MPC/NMPC、轨迹跟踪、动态障碍与重规划通知 | [MPC 动态避障](src/docs/simulation-and-planning/mpc动态避障.md) |

## 演示视频

下表只说明视频展示的场景，不把仿真或部署演示等同于真实机械臂验收。

| 功能 | 视频 | 证据边界 | 关联文档 |
| --- | --- | --- | --- |
| MPC 动态避障 | [观看视频](https://www.bilibili.com/video/BV1mmeE6QERY) | Gazebo 仿真 | [MPC 动态避障](src/docs/simulation-and-planning/mpc动态避障.md) |
| 机械臂视觉伺服 | [观看视频](https://www.bilibili.com/video/BV1zNeJ6gEYX) | 部署演示 | [位置伺服](src/docs/simulation-and-planning/visual-servo-simulation.md) |
| 机械臂视觉抓取 | [观看视频](https://www.bilibili.com/video/BV1NUej65Emf) | 演示视频 | [YOLO 视觉抓取](src/docs/perception-and-grasping/yolov8-visual-grasping.md) |
| 指挥我的机械臂帮我干活！LLM-Control | [观看视频](https://www.bilibili.com/video/BV1Yeam6TEVo) | LLM-Control 演示 | [LLM 控制](src/docs/perception-and-grasping/llm-yolo-control.md) |
| GraspNet 机械臂部署 | [观看视频](https://www.bilibili.com/video/BV1hQeV6yEfk) | 真实机械臂部署演示 | [GraspNet 抓取](src/docs/perception-and-grasping/graspnet-simulation.md) |

## 环境与依赖

- Ubuntu 22.04 与 ROS 2 Humble；
- MoveIt 2、ros2_control、Gazebo Fortress（Ignition）与 `ros_gz_bridge`；
- Python 3、NumPy/SciPy、OpenCV，以及各功能包声明的 ROS 依赖；
- 视觉功能需要对应相机、模型文件和标定结果；
- LLM 语音功能还需要音频依赖、包内 KWS 模型和有效的 Qwen/DashScope 凭据；
- GraspNet 与 MPC 分别有独立的深度学习环境和 acados/CasADi 依赖。

仓库包含多个上游组件和本地模型资产，不能假设一次通用 `rosdep` 命令即可配置所有外部运行时。首次复现前请先阅读目标功能的专题文档。

## 快速运行

在工作区根目录执行：

```bash
source /opt/ros/humble/setup.bash
./run_build.sh
source install/setup.bash
ros2 launch myrobot_simulation gazebo.launch.py
```

`run_build.sh` 按本机工作区策略跳过相机驱动、`realsense2_gz_description` 和 `fairino_hardware`。首次克隆的环境必须先安装或单独构建这些依赖；不要把脚本完成等同于全部硬件依赖已经就绪。

常用业务入口：

```bash
# 路径规划与 IK 对比
ros2 launch myrobot_simulation motion_planning_demo_sim.launch.py

# YOLO 视觉抓取仿真
ros2 launch myrobot_simulation visual_grasping_sim.launch.py

# 位置视觉伺服仿真
ros2 launch myrobot_simulation visual_position_servo_sim.launch.py

# LLM 实时语音控制仿真
ros2 launch myrobot_simulation llm_robot_control_sim.launch.py
```

真实机械臂、GraspNet、手眼标定和 MPC 的依赖与安全条件不同，请使用对应专题文档中的入口，不要直接照搬仿真命令。

## 目录结构

```text
fairino_robotarm/
├── src/docs/                         # 技术文档中心
├── src/myrobot_planning_core/        # ROS 无关的规划与解析 IK 核心
├── src/myrobot_planning_ros/         # MoveIt 规划器与 IK 插件
├── src/myrobot_common_ws/            # 运动、感知、取消与轨迹公共能力
├── src/myrobot_simulation/           # Gazebo 与业务仿真入口
├── src/visual_perception/            # YOLO RGB-D 感知
├── src/visual_grasping_bringup/      # YOLO 抓放状态机
├── src/visual_servo_bringup/         # 图像/位置视觉伺服
├── src/llm_arm_control/              # KWS、Qwen 与受限任务执行
├── src/graspnet_ws/                  # GraspNet 推理与执行
├── src/calibration_ws/               # ArUco 与手眼标定
├── src/camera_ws/                    # RealSense/OAK 驱动和仿真描述
├── src/myrobot_mpc_ws/               # MPC/NMPC 动态避障
└── src/myrobot_support_ws/           # Fairino 模型、MoveIt 配置和硬件接口
```

## 阅读路线

1. [技术文档中心](src/docs/README.md)：先建立工作区全貌。
2. [仿真环境架构](src/docs/simulation-and-planning/仿真环境架构说明.md)：理解 Gazebo、MoveIt 和业务 Launch 的关系。
3. 按目标选择[规划](src/docs/simulation-and-planning/planning-demo.md)、[抓取](src/docs/perception-and-grasping/yolov8-visual-grasping.md)、[LLM](src/docs/perception-and-grasping/llm-yolo-control.md)或[标定](src/docs/手眼标定/手眼标定文档说明.md)专题。
4. 完成静态配置和仿真检查后，再进入真实机械臂流程。

## 安全与公开边界

- 软件停止、Action 取消或状态机锁存不是安全等级急停；实机必须具备独立硬件安全链路。
- 仓库根目录目前没有统一许可证；各包和第三方组件按各自声明管理，不能据此推定整个仓库可按同一许可证再分发。
- KWS、YOLO、GraspNet 等模型可能具有独立许可或非商业限制，提交或再分发前必须单独核验。
- API 凭据、设备序列号、个人路径、标定结果和运行日志不应直接提交到公开仓库。
