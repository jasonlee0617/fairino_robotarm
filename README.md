# Fairino Robot Arm

这是 Fairino 六轴机械臂 ROS 2 工作区的项目入口：整合 `ROS 2 Humble`、`MoveIt 2`、Gazebo、自定义采样规划、RGB-D 感知、抓取、视觉伺服、LLM 任务控制、手眼标定与 MPC 动态避障。本文用于快速了解项目能力、观看演示和开始复现；详细实现说明请进入技术文档中心。

> 当前公开证据边界：本仓库优先展示代码、文档和 Gazebo 仿真流程。真实机械臂结果会在完成独立安全检查与录制后单独标注，不能由仿真画面替代。

[技术文档](src/docs/README.md) · [演示录制与发布指南](src/docs/演示视频录制与发布指南.md) · [快速运行](#快速运行)

## 项目亮点

| 能力 | 项目内容 | 当前公开证据 |
| --- | --- | --- |
| 运动规划与 IK | Fairino 自定义 RRT 系列、MoveIt 规划管线、解析 IK 与诊断 | [规划说明](src/docs/simulation-and-planning/规划算法结构说明.md) |
| 感知与抓取 | RGB-D、YOLO、GraspNet 和抓取执行链路 | [感知与抓取说明](src/docs/perception-and-grasping) |
| LLM 任务控制 | 自然语言、目标消歧、预览确认与安全状态机 | [LLM 控制说明](src/docs/perception-and-grasping/llm-yolo-control.md) |
| 标定与视觉伺服 | ArUco 手眼标定、Eye-in-Hand、Eye-on-Base、Gazebo 视觉伺服 | [手眼标定说明](src/docs/手眼标定/手眼标定文档说明.md) |

## 演示与证据

下表链接至已发布的 Bilibili 演示视频。请结合“证据边界”理解演示结果：仿真、部署演示和真实机械臂结果不互相替代。

| 功能 | 演示说明 | 视频 | 证据边界 | 关联文档 |
| --- | --- | --- | --- | --- |
| MPC 动态避障 | MPC 在动态障碍场景中的避障规划与执行 | [观看视频](https://www.bilibili.com/video/BV1mmeE6QERY?vd_source=45556d309c289d529d67646995b5219f) | Gazebo 仿真 | [MPC 动态避障](src/docs/simulation-and-planning/mpc动态避障.md) |
| 机械臂视觉伺服 | 视觉伺服落地部署流程演示 | [观看视频](https://www.bilibili.com/video/BV1zNeJ6gEYX?vd_source=45556d309c289d529d67646995b5219f) | 部署演示 | [视觉伺服](src/docs/simulation-and-planning/visual-servo-simulation.md) |
| 机械臂视觉抓取 | 视觉感知、目标识别与抓取流程演示 | [观看视频](https://www.bilibili.com/video/BV1NUej65Emf?vd_source=45556d309c289d529d67646995b5219f) | 演示视频 | [YOLO 视觉抓取](src/docs/perception-and-grasping/yolov8-visual-grasping.md) |
| LLM-DeepSeek 机械臂控制抓取 | 自然语言任务控制与抓取流程演示 | [观看视频](https://www.bilibili.com/video/BV1zNeJ6gEBw?vd_source=45556d309c289d529d67646995b5219f) | 演示视频 | [LLM 任务控制](src/docs/perception-and-grasping/llm-yolo-control.md) |
| GraspNet 机械臂部署 | GraspNet 抓取在真实机械臂上的部署演示 | [观看视频](https://www.bilibili.com/video/BV1hQeV6yEfk?vd_source=45556d309c289d529d67646995b5219f) | 真实机械臂部署 | [GraspNet 抓取](src/docs/perception-and-grasping/graspnet-simulation.md) |


## 文档与复现

- [工作区架构与包职责](src/docs/README.md)
- [仿真与规划](src/docs/simulation-and-planning/)
- [感知、抓取与 LLM 控制](src/docs/perception-and-grasping/)
- [手眼标定](src/docs/手眼标定/)
- [MPC 动态避障](src/docs/simulation-and-planning/mpc动态避障.md)

