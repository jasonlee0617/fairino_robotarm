# 相机模型在 Ignition Gazebo 6 中正常显示指南

## 问题

Intel RealSense D435 相机模型在 RViz 中正常显示，但在 Ignition Gazebo 6 中不显示（或仅显示灰白色无纹理）。

## 根因

| 层次 | 问题 | 后果 |
|---|---|---|
| 1. `package://` 路径 | `robot_description_with_package_paths()` 只解析了 `fairino_arm_moveit_descriptions` 包路径 | Gazebo 收不到 mesh 文件系统路径 |
| 2. URI scheme | 解析后路径无 `file://` 前缀 | Ignition Common 当作相对路径在 `GZ_SIM_RESOURCE_PATH` 中搜索 → 搜不到 |
| 3. GZ_SIM_RESOURCE_PATH | `realsense2_description` 不在搜索路径中 | 兜底搜索也失败 |
| 4. **COLLADA 材质与 Ogre2 不兼容** | Ogre2 无法解析 `.dae` 文件的 `<lambert>` 材质定义 | mesh 几何体加载了，但 18 个子 mesh 全部不可见 |
| 5. 强制灰色覆盖 | `<gazebo><material>Gazebo/Grey</material></gazebo>` 覆盖了所有材质 | 即使 mesh 能渲染，也被染成浅灰色 |

**铁证**在 `/home/robot/.ignition/rendering/ogre2.log`：
```
Can't assign material scene::Material(65497) because this Material does not exist.
Have you forgotten to define it in a .material script?
```
此错误重复 79 条，对应 D435.dae 的 18 个子 mesh 多次绑定失败。

## 解决方案

### 1. 转换 COLLADA → STL（解决材质兼容性）

STL 格式无材质定义，Ogre2 用默认材质直接渲染。

```bash
# 安装 trimesh
pip3 install trimesh[easy] --user

# 转换（DAE scene → 合并所有几何体 → 单个 STL）
python3 -c "
import trimesh
scene = trimesh.load('/opt/ros/humble/share/realsense2_description/meshes/d435.dae')
meshes = [g for g in scene.geometry.values() if hasattr(g, 'vertices')]
combined = trimesh.util.concatenate(meshes)
combined.export('$(ros2 pkg prefix fairino_arm_moveit_descriptions)/share/fairino_arm_moveit_descriptions/meshes/d435.stl')
print(f'Done: {len(combined.vertices)} vertices, {len(combined.faces)} faces')
"
# 输出: Done: 154149 vertices, 231186 faces (12 MB)
```

### 2. 修改 `moveit_stack.py` — 解析所有 `package://` 引用并加 `file://` 前缀

**文件**: `myrobot_simulation/launch_utils/moveit_stack.py`

```python
def robot_description_with_package_paths(moveit_config, profile: RobotProfile) -> str:
    """Expand ALL package:// references to filesystem paths for ros_gz_sim."""
    import re
    description = moveit_config.robot_description["robot_description"]

    def resolve(m):
        pkg = m.group(1)
        try:
            return "file://" + get_package_share_directory(pkg)
        except Exception:
            return m.group(0)

    return re.sub(r'package://([^/]+)', resolve, description)
```

关键变更：
- 原来：`return get_package_share_directory(pkg)` → 路径无 scheme
- 现在：`return "file://" + get_package_share_directory(pkg)` → `file:///opt/ros/humble/share/...`
- Ignition Common 通过 `file://` scheme 直接读文件系统

### 3. 修改 `sim_stack.py` — 把 `realsense2_description` 加入资源路径

**文件**: `myrobot_simulation/launch_utils/sim_stack.py`

```python
def sim_resource_path(profile: RobotProfile):
    gz_share = get_package_share_directory("myrobot_simulation")
    desc_share = get_package_share_directory(profile.description_package)
    try:
        realsense_share = get_package_share_directory("realsense2_description")
    except Exception:
        realsense_share = None
    paths = [
        os.path.join(gz_share, "worlds"),
        os.path.join(gz_share, "worlds", "models"),
        str(Path(desc_share).parent.resolve()),
    ]
    if realsense_share:
        paths.append(str(Path(realsense_share).resolve()))
    return SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=":".join(paths),
    )
```

### 4. 修改 `_d435.urdf.xacro` — 使用本地 STL + 真实材质颜色

**文件**: `fairino_arm_moveit_descriptions/urdf/camera/_d435.urdf.xacro`

```xml
<!-- Mesh 引用改为本地 STL（第 78 行） -->
<mesh filename="package://fairino_arm_moveit_descriptions/meshes/d435.stl" />

<!-- 添加真实 D435 深灰/黑色（visual 内新增） -->
<material name="DarkGrey">
  <color rgba="0.16 0.16 0.16 1.0"/>
</material>
```

### 5. 修改 `camera.xacro` — 使用本地宏 + 去掉强制灰色

**文件**: `fairino_arm_moveit_descriptions/urdf/camera/camera.xacro`

```xml
<!-- Include 从系统包改为本地宏 -->
<!-- <xacro:include filename="$(find realsense2_description)/urdf/_d435.urdf.xacro" /> -->
<xacro:include filename="$(find fairino_arm_moveit_descriptions)/urdf/camera/_d435.urdf.xacro" />

<!-- 删除强制灰色覆盖：<material>Gazebo/Grey</material> -->
```

## 文件变更汇总

| 文件 | 改动 |
|---|---|
| `fairino_arm_moveit_descriptions/meshes/d435.stl` | **新建** — DAE → STL 转换，12MB |
| `fairino_arm_moveit_descriptions/urdf/camera/_d435.urdf.xacro` | mesh 改为本地 STL；新增 `<material>` |
| `fairino_arm_moveit_descriptions/urdf/camera/camera.xacro` | include 从系统包改为本地；删除 `Gazebo/Grey` |
| `myrobot_simulation/launch_utils/moveit_stack.py` | `package://` 解析后加 `file://` 前缀 |
| `myrobot_simulation/launch_utils/sim_stack.py` | GZ_SIM_RESOURCE_PATH 加入 `realsense2_description` |

## 为什么 RViz 正常但 Gazebo 不行

| | RViz | Ignition Gazebo 6 |
|---|---|---|
| 渲染引擎 | RViz 内置 (OGRE 1.x) | Ogre2 |
| `package://` 解析 | ament index（全局 ROS 包） | 有限，需要 `file://` 或 `GZ_SIM_RESOURCE_PATH` |
| COLLADA 材质 | 内置支持 | **不支持**，只能用 STL/OBJ |
| `<material>` 覆盖 | 无此概念 | `<gazebo>` 标签可覆盖 URDF visual |

## 添加新相机 mesh 的通用流程

如果将来需要添加其他相机模型（OAK-D 等），按以下步骤：

1. 获取 mesh 源文件（.dae / .glb / .obj）
2. 转换为 STL：`trimesh.load('source').export('output.stl')`
3. 放入 `fairino_arm_moveit_descriptions/meshes/`
4. 在 URDF 中引用：`<mesh filename="package://fairino_arm_moveit_descriptions/meshes/xxx.stl"/>`
5. 在 `<visual>` 中加 `<material><color rgba="..."/></material>`
6. **不要**在 `<gazebo>` 标签中加 `<material>` 覆盖
