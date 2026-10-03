# Fairino ROS 2 硬件接口

该包基于 Fairino SDK 提供：

- ros2_control `SystemInterface` 硬件插件；
- `/FR_ROS_API_service` 字符串命令服务；
- 示例客户端和厂商 SDK 运行库。

常用字符串命令见[API 说明](API说明.md)，完整可调度命令以
`include/fairino_hardware/command_server.hpp` 的 `_fr_function_list` 为准。
上层工作区架构见[技术文档中心](../../docs/README.md)。

厂商安装与控制器说明请参考
[Fairino ROS 2 文档](https://fr-documentation.readthedocs.io/zh_CN/latest/ROSGuide/ros2guide.html)。

> [!WARNING]
> 字符串服务会调用真实机器人 SDK。实机发送运动、使能、I/O 或安全参数命令前，
> 必须确认控制器模式、坐标系、负载、速度、限位和硬件急停；服务返回成功不能替代现场安全检查。
