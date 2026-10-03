# D435 hardware profiles

These profiles are consumed by the
[Fairino simulation stack](../../../../docs/simulation-and-planning/仿真环境架构说明.md).

This directory contains canonical, per-resolution D435 `CameraInfo` profiles
used by the Fairino Gazebo camera model. They are hardware/profile data, not
hand-eye calibration results or device archives.

Capture a profile with:

```bash
ros2 run hand_eye_calibration capture_d435_profile.py --ros-args \
  -p color_profile:=<color-width>x<color-height>x<fps> \
  -p depth_profile:=<depth-width>x<depth-height>x<fps>
```

例如采集真实 `640x480x60` 的彩色与深度流，先只启动相机：

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  align_depth.enable:=true
```

再另开终端执行：

```bash
ros2 run hand_eye_calibration capture_d435_profile.py --ros-args \
  -p color_profile:=640x480x60 \
  -p depth_profile:=640x480x60 \
  -p sample_count:=60
```

采集器会验证实际 Color/Depth 图像尺寸与帧率；不匹配、重复时间戳或深度 ROI
无有效数据时拒绝覆盖文件。采集时相机与 0.5--1.0 m 处的平整目标面必须静止。

该脚本会原子性地替换所选彩色/深度配置对应的标准文件，并在 captured_at 字段中记录采集时间。
在 Gazebo 中使用标准配置文件时，通过 camera_profile:=<文件名主干> 指定，例如
camera_profile:=d435_color_640x480x30_depth_640x480x30。
仅当需要临时外部配置文件时才使用 camera_profile_file:=<yaml 文件>。
camera_profile 与 camera_profile_file 互斥；对于默认使用命名配置文件的入口节点，若要改用外部文件，需传入 camera_profile:='' 并同时指定外部文件。

文件名中的 x30/x60 后缀记录真实采集模式。仿真应默认使用同一 FPS；如需不同帧率，
必须在 launch 中显式传入 camera_fps，并确认该 D435 模式支持该帧率。
