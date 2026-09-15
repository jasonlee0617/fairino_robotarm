# LLM-YOLO-Control 工作原理

## 概述

LLM-YOLO-Control 是一个将**大语言模型（DeepSeek）**与**YOLO 视觉检测**结合、实现**自然语言驱动机器人操作**的系统。用户用自然语言描述任务（如"把红色螺丝刀放到左边的盒子里"），系统自动理解场景、规划动作序列并执行。

**核心思路：LLM 不做视觉理解，而是做结构化数据的数值推理。**

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       用户自然语言指令                             │
│                  "抓取最左边的螺丝钉放到盒子里"                      │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                    LlmControlTaskServer                           │
│                  (llm_control_task_server.py)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────────┐   ┌───────────────────┐  │
│  │ 视觉感知层   │   │   LLM 推理层     │   │   运动执行层       │  │
│  │             │   │                 │   │                   │  │
│  │ YOLO OBB   │   │ DeepSeek API    │   │ MoveIt2 +        │  │
│  │ + 深度估计  │──▶│ chat/completions│──▶│ fairino_planning │  │
│  │ + TF变换   │   │                 │   │ _core (BiRRT*)    │  │
│  │             │   │ 文本指令→JSON   │   │                   │  │
│  └─────────────┘   └─────────────────┘   └───────────────────┘  │
│                                                                  │
│  输入: RGB图 + 深度图         输出: 机器人关节轨迹 → 硬件执行       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 第一阶段：视觉感知（从像素到 3D 世界坐标）

### 1.1 数据来源

| 话题 | 来源包 | 内容 |
|------|--------|------|
| `/yolo/detected_result` | `visual_perception/llm_visual_perception.py` | YOLOv8 OBB 检测结果：class_name, confidence, 4 角点像素坐标 (u,v) |
| `/yolo/detected_result/depth` | `llm_visual_perception.py` | 对齐的深度图像 (16UC1 或 32FC1) |
| `/camera/.../camera_info` | 相机驱动 | 内参矩阵 K (fx, fy, cx, cy) |

### 1.2 RGB-D 时间同步

YOLO 帧和深度帧可能不是同一时刻采集的，因此需要配对：

```
YOLO 帧缓冲区 (最多 20 帧)  ──┐
                             ├── 遍历所有 YOLO×Depth 组合
深度帧缓冲区 (最多 20 帧)  ──┘   找时间戳差 < 50ms 的最新的一对
                                        │
                                        ▼
                                  active_frame (配对后的帧)
```

如果超过 1 秒没有新鲜配对帧，认为视觉输入不可用。

### 1.3 从 2D 像素到 3D 基座坐标（关键步骤）

对 YOLO 检测到的每个物体，执行以下变换链：

```
步骤 1: 从 OBB 四个角点提取中心像素坐标
        center_uv = mean(points[0:4])

步骤 2: 鲁棒深度估计 (robust_center3d_from_obb_depth)
        - 在 OBB 多边形区域内采样深度值（最多 5000 个点）
        - 使用 MAD (Median Absolute Deviation) 剔除异常值
        - 要求深度内点率 ≥ 60%
        - 输出: 相机坐标系下的 3D 位置 (x_c, y_c, z_c)
                                        │
步骤 3: 计算物体 yaw（偏航角）
        - 取 OBB 的最长边作为主轴方向
        - 将主轴方向投影到 3D 空间
        - 通过 TF 变换到 base_link 坐标系
        - yaw = atan2(dy, dx)，归一化到 [-π/2, π/2]
                                        │
步骤 4: TF 坐标变换 (camera_frame → base_link)
        - 对步骤 2 的 (x_c, y_c, z_c) 做 TF 变换
        - 输出: base_link 坐标系下的 3D 位置 (x_b, y_b, z_b)
                                        │
步骤 5: 组装 ResolvedCandidate
        {
          index: 0,
          class_name: "elongated_object",
          confidence: 0.92,
          center_uv: (320, 240),     ← 像素坐标，用于"左/右"推理
          base_xyz: (0.35, -0.12, 0.05), ← 基座坐标，用于"近/远"推理
          yaw: 0.23,                 ← 偏航角，用于确定夹取方向
          depth_inlier_ratio: 0.85   ← 深度质量指标
        }
```

**为什么需要两套坐标？**
- `center_uv`（像素坐标）：LLM 用来推理 **左/右** 位置关系
- `base_xyz`（基座坐标）：LLM 用来推理 **远/近** 距离关系、系统用来计算运动位姿

---

## 第二阶段：LLM 推理（从自然语言到结构化任务计划）

### 2.1 LLM 的角色定位

**LLM 不是视觉模型，不理解图像。** 它是一个文本→JSON 翻译器，接收包含精确数值的结构化上下文，输出结构化的动作序列 JSON。

这是整个系统最核心的设计理念：

> 让 LLM 做它擅长的事（数值比较和逻辑推理），让视觉模块做它擅长的事（从像素中提取结构化信息），两者通过精心设计的 JSON 接口连接。

### 2.2 指令预处理

```python
# 步骤 1: 判断是否有视觉意图
instruction_has_visual_intent(instruction)
# 正则匹配: 抓|夹取|拿|拾取|放|摆放|pick|grasp|place|put
# 有视觉意图 → 等待最新检测帧 + 3D 位姿（超时 15 秒）
# 无视觉意图 → 使用缓存元数据（如纯移动指令 "向左移动 5cm"）

# 步骤 2: 本地确定性选择视觉目标
# bolt/螺栓/螺丝/pen/笔 → elongated_object；左/右/上/下/中间/最近/最远
# 有明确筛选词 → 直接选择；同类多目标且无筛选词 → 请求澄清

# 步骤 3: 检查是否有消歧词
_has_disambiguator(instruction)
# 关键词列表: 左,右,前,后,上,下,最近,最远,靠近,第,编号,索引
#              left,right,front,back,nearest,farthest,index,number
# 有消歧词 → reject_ambiguous=False (允许同类物体选一个)
# 无消歧词 → reject_ambiguous=True  (同类多个则要求用户澄清)
```

### 2.3 构建 LLM 上下文

将视觉感知结果 + 用户指令 + 机器人状态打包为一个 JSON：

```json
{
  "instruction": "抓取最左边的螺丝钉",
  "candidates": [
    {
      "index": 0,
      "class_name": "elongated_object",
      "confidence": 0.92,
      "center_uv": [120, 245],
      "base_xyz": [0.25, -0.10, 0.05],
      "yaw": 0.12,
      "depth_inlier_ratio": 0.85
    },
    {
      "index": 1,
      "class_name": "elongated_object",
      "confidence": 0.88,
      "center_uv": [310, 250],
      "base_xyz": [0.35, -0.08, 0.05],
      "yaw": 0.18,
      "depth_inlier_ratio": 0.91
    },
    {
      "index": 2,
      "class_name": "elongated_object",
      "confidence": 0.95,
      "center_uv": [520, 242],
      "base_xyz": [0.45, -0.11, 0.04],
      "yaw": -0.05,
      "depth_inlier_ratio": 0.88
    },
    {
      "index": 3,
      "class_name": "box",
      "confidence": 0.90,
      "center_uv": [400, 300],
      "base_xyz": [0.40, 0.20, 0.03],
      "yaw": -0.15,
      "depth_inlier_ratio": 0.93
    }
  ],
  "holding_class": null,
  "current_pose": {
    "frame_id": "base_link",
    "x": 0.30, "y": 0.0, "z": 0.50,
    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0
  }
}
```

### 2.4 系统提示词（System Prompt）设计

System Prompt 是整个系统的核心，它精确规定了 LLM 的输出格式和推理规则：

```python
SYSTEM_PROMPT = """你通过一个经过严格验证的本地规划器来控制 Fairino 机械臂。
请仅返回 JSON 格式，且必须包含且仅包含一个顶层键：{"actions": [...]}。
允许的动作对象如下：
1. {"type":"pick","source_index":int}
2. {"type":"place","destination_index":int}
3. {"type":"pick_place","source_index":int,"destination_index":int}
4. {"type":"move_relative","dx":m,"dy":m,"dz":m,...}
5. {"type":"move_absolute","x":m,"y":m,"z":m,"qx","qy","qz","qw",...}
6. {"type":"set_gripper","state":"open|close"}
7. {"type":"home"}

视觉类别 elongated_object 包含语言别名 pen 和 bolt。

当请求抓取但未指定目标位置时，使用 pick；
当已持有物体并请求放置时，使用 place；
当同时指定了源位置和目标位置时，使用 pick_place。

绝不能用 set_gripper、move_relative 或 move_absolute 来替代视觉抓取/放置请求。

只能使用已列出的检测索引，严禁捏造视觉坐标。
如果请求存在歧义，不要猜测：请返回一个包含无效索引的动作，以便本地验证器将其拒绝。

候选目标的 center_uv 单位为图像像素：最左侧的 u 值最小，最右侧的 u 值最大。
候选目标的 base_xyz 基于 base_link 坐标系。对于最近/最远的请求，请比较从 base_xyz 到 current_pose 的欧几里得距离。

对于视觉任务，必须且只能返回一个 pick、place 或 pick_place 动作。
只有当 holding_class 不为 null 时，place 动作才有效。

最多返回八个动作。"""
```

**System Prompt 的设计要点：**

| 设计原则 | 具体体现 |
|----------|----------|
| **输出约束** | 只能输出 JSON `{"actions": [...]}`，不能输出解释性文本 |
| **动作类型白名单** | 只有 7 种动作类型，每种都有固定的字段 Schema |
| **视觉安全** | 不能将视觉任务降级为夹爪操作或盲移动 |
| **坐标来源** | LLM 只能用提供的 index，禁止自创坐标 |
| **空间推理指引** | 明确告诉 LLM "leftmost = 最小 u", "nearest = 最小距离" |
| **失败策略** | 不确定时返回无效 index 让本地验证器拒绝，而非猜测 |
| **单视觉动作** | 一个 plan 只能有一个视觉动作，防止 LLM 生成危险的多步序列 |

### 2.5 多轮对话与语义历史

系统维护会话历史，但做了关键的**语义压缩**：

```
原始 LLM 回复:
{"actions":[{"type":"pick_place","source_index":0,"destination_index":3}]}

压缩后的历史:
{"actions":[{"type":"pick_place","source_class":"elongated_object","destination_class":"box"}]}
```

**为什么要压缩？**
- 具体的 `source_index: 0` 在下一轮对话时已经过期（物体可能移动、检测索引会变）
- 但语义信息 `source_class: elongated_object` 在多轮对话中有用（如："再拿一个同样的"）
- 历史最多保留 20 条消息

### 2.6 LLM 如何"理解"空间描述

LLM 不接收图像。它通过**比较 JSON 中的数值**来推理空间关系。

#### "最左边" → 比较 center_uv[0] (u 像素坐标)

```
candidates 的 u 坐标:  [120, 310, 520]
                        ↑
                    最小 u = 图像最左边 = index 0
```

LLM 依据系统提示词（System Prompt）中的 "leftmost has the smallest u"，选出 u 坐标最小的候选目标。

#### "最右边" → 比较 center_uv[0]

```
candidates 的 u 坐标:  [120, 310, 520]
                                ↑
                    最大 u = 图像最右边 = index 2
```

#### "最近" / "最远" → 欧几里得距离

```
current_pose: (0.30, 0.0, 0.50)

index 0: base_xyz=(0.25, -0.10, 0.05) → dist = √((0.30-0.25)²+(0-(-0.10))²+(0.50-0.05)²) = 0.464
index 1: base_xyz=(0.35, -0.08, 0.05) → dist = √((0.30-0.35)²+(0-(-0.08))²+(0.50-0.05)²) = 0.461 ← 最近
index 2: base_xyz=(0.45, -0.11, 0.04) → dist = √((0.30-0.45)²+(0-(-0.11))²+(0.50-0.04)²) = 0.498
```

LLM 计算每个 candidate 的 `base_xyz` 到 `current_pose` 的欧几里得距离，选最小/最大的。

#### "前面" / "后面" → 比较 base_xyz[0] (x 坐标)

```
candidates 的 x 坐标: [0.25, 0.35, 0.45]
                                ↑
                  current_pose.x=0.30, 最近且 >0.30 的 = index 1 是"前面"
```

#### "上面" / "下面" → 比较 base_xyz[2] (z 坐标)

```
candidates 的 z 坐标: [0.05, 0.05, 0.04]
                      最大 z = index 0 或 1 是"上面"
```

#### "中间" → 中位数推理

虽然 System Prompt 没有明确提及"中间"，但 LLM 具备数值推理能力：
- 计算所有候选 u 坐标的中位数
- 选出 u 最接近中位数的候选

### 2.7 消歧安全机制

```
                    用户输入指令
                         │
            是否有消歧词?(左/右/前/后/最近/最远/第/编号)?
                  │                    │
                 YES                  NO
                  │                    │
          reject_ambiguous     reject_ambiguous
              = False              = True
                  │                    │
          LLM 可自由选择        同类物体 > 1 个?
          一个候选               │         │
                              YES        NO
                               │         │
                        拒绝并要求      LLM 选择
                        用户澄清      唯一候选 → 通过
```

**关键点：`"中间"` 不在消歧词列表中。**

| 用户指令 | 同类物体数 | 有消歧词? | 结果 |
|----------|-----------|----------|------|
| "抓取最左边的螺丝钉" | 3 | ✅ 有"左" | LLM 选 index 0 → **通过** |
| "抓取中间的螺丝钉" | 3 | ❌ "中间"不在列表 | LLM 选 index 1 → **被拒绝** |
| "抓取螺丝钉" | 1 | N/A | LLM 选 index 0 → **通过** |
| "抓取螺丝钉" | 3 | ❌ 无 | **被拒绝** → 要求澄清 |

**这是设计上的保守策略：** "中间"确实有歧义——是图像左右方向的中间还是机器人前后方向的中间？系统宁可拒绝执行也不冒险猜测。

### 2.8 DeepSeek API 调用

```python
# deepseek_client.py
def chat(self, messages, model="deepseek-chat"):
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,  # 多轮对话压缩历史
            {"role": "user", "content": context_json},
        ],
        "response_format": {"type": "json_object"},  # 强制 JSON 输出
        "stream": False,
    }
    # POST https://api.deepseek.com/chat/completions
    # timeout: 30s
```

- 使用 `response_format: json_object` 强制 LLM 输出合法 JSON
- API Key 通过 GNOME Keyring 管理 (`deepseek_credentials.py`)
- 30 秒超时

---

## 第三阶段：计划验证

LLM 返回的 JSON 不是直接执行的，而是经过严格的多层验证。

### 3.1 JSON Schema 验证

```python
def parse_llm_plan(text, candidates, ...):
    # 1. 必须是合法 JSON
    data = json.loads(text)

    # 2. 只能有 {"actions": [...]} 一个顶层键
    if set(data) != {"actions"}: raise ValueError

    # 3. actions 必须在 1-8 个之间
    if not 1 <= len(actions) <= 8: raise ValueError

    # 4. 每个 action 的字段必须完全匹配 Schema
    #    pick: {"type","source_index"}
    #    pick_place: {"type","source_index","destination_index"}
    #    move_relative: {"type","dx","dy","dz","droll_deg","dpitch_deg","dyaw_deg","frame_id"}
    #    ...
```

### 3.2 Candidate 验证

```python
# source_index 必须是候选列表中的索引
# source 的 class_name 必须在 pick_classes (elongated_object, cube) 中
# destination 的 class_name 必须在 place_classes (box) 中

if selected.class_name not in allowed_classes:
    raise ValueError(f"class {name!r} is not pickable")

# pick_place 的 source 和 destination 不能相同
if source_index == destination_index:
    raise ValueError("source and destination must differ")
```

### 3.3 Intent 验证

```python
# 如果用户指令包含视觉意图词(抓/放/pick/place...)
# 但 LLM 返回的 plan 没有视觉动作(pick/place/pick_place)
# → 拒绝。禁止 LLM 将视觉任务降级为 set_gripper 或 move_relative
```

### 3.4 状态机验证

```python
# 已持有物体时，只能 place，不能 pick
# 未持有时，只能 pick，不能 place
# 放置失败后保持 HOLDING；请创建新的确认预览。
```

---

## 第四阶段：动作富化（将抽象计划变为可执行的 3D 位姿序列）

LLM 的 JSON 只包含 `{"type":"pick","source_index":0}`，需要将其"富化"为包含 3D 位姿的完整步骤序列。

### 4.1 抓取高度计算

有两种模式（由 `use_visual_z` 参数控制）：

**固定高度模式（默认）：**
```
grasp_z  = 0.02m    (桌面以上 2cm)
approach = 0.12m    (抓取点以上 10cm)
carry    = 0.15m    (搬运高度)
```

**视觉高度模式：**
```
grasp_z  = 物体Z + visual_grasp_offset_z   (基于检测到的物体实际高度)
approach = grasp_z + 0.10m
carry    = grasp_z + 0.13m
```

### 4.2 抓取姿态计算

```python
def _grasp_quat(self, yaw):
    # roll=0°, pitch=-180° (夹爪朝下)
    # yaw = 物体偏航角 + π/2 (从物体最长边的侧面夹取)
    return Rotation.from_euler(
        "xyz",
        [0.0, -180.0, math.degrees(yaw + math.pi/2)],
        degrees=True
    ).as_quat()
```

### 4.3 动作展开示例

**LLM 输出：** `{"type":"pick_place","source_index":0,"destination_index":3}`

**展开为 13 个具体步骤：**

```
PICK 阶段 (6步):
┌─────┬──────────────────┬────────────┬──────────────────────────┐
│ 步  │ 操作             │ 运动模式    │ 说明                     │
├─────┼──────────────────┼────────────┼──────────────────────────┤
│  1  │ open_gripper     │ -          │ 打开夹爪到 0.061m         │
│  2  │ go_home          │ Joint(BiRRT*)│ 回到预定义安全关节角    │
│  3  │ approach_pick    │ Joint(BiRRT*)│ 到抓取点上方, 0.5x 速度 │
│  4  │ grasp            │ Cartesian   │ 直线下降到抓取高度,0.2x │
│  5  │ close_gripper    │ -           │ 闭合夹爪                  │
│  6  │ carry            │ Cartesian   │ 直线提升到搬运高度,0.2x  │
├─────┼──────────────────┼────────────┼──────────────────────────┤
│PLACE 阶段 (7步):                                                    │
├─────┼──────────────────┼────────────┼──────────────────────────┤
│  7  │ re_detect_box    │ -          │ 采集5帧盒子位置+稳定性验证│
│  8  │ approach_box     │ Joint(BiRRT*)│ 到盒子安全高度, 0.5x    │
│  9  │ release          │ Cartesian   │ 直线下降到释放高度,0.2x  │
│ 10  │ release_gripper  │ -           │ 打开夹爪释放              │
│ 11  │ box_retreat      │ Cartesian   │ 直线后退到安全高度,0.5x  │
│ 12  │ return_home      │ Joint(BiRRT*)│ 回到 home                │
│ 13  │ close_gripper    │ -           │ 闭合夹爪(回初始状态)      │
└─────┴──────────────────┴────────────┴──────────────────────────┘
```

**为什么 grasp/carry/release 必须用 Cartesian 直线运动？**
- 抓取和释放阶段，夹爪在物体附近，用 Joint space 规划可能走出弧线路径，碰撞到物体侧面
- Cartesian 直线确保末端沿 Z 轴垂直上升/下降。

### 4.4 安全性检查

每个计算出的位姿在加入计划前，都要通过：

```python
def _check_pose(self, pose):
    # 1. 工作空间范围检查
    #    X: [-0.10, 0.60], Y: [-0.60, 0.60], Z: [0.01, 0.55]
    if not workspace_ok(xyz): raise ValueError

    # 2. 碰撞感知 IK 检查 (通过 MoveIt2 compute_ik)
    if moveit2_arm.compute_ik(xyz, quat) is None:
        raise ValueError("no collision-aware IK solution")
```

---

## 第五阶段：预览-确认-执行

### 5.1 两阶段提交协议

系统采用"预览-确认"模式，防止 LLM 幻觉导致的误操作：

```
阶段 1: PreviewCommand 服务 (/llm_control/preview_command)
  - 生成完整的执行计划
  - 计算所有 3D 位姿
  - 验证所有 IK 可行性
  - 返回详细的 JSON 预览给用户
  - 状态: IDLE → PREVIEW_READY
  - 有效期: 15 秒

阶段 2: ExecutePreview Action (/llm_control/execute_preview)
  - 用户输入 y 确认
  - 重新验证物体位置（可能在预览期间移动）
  - 逐步执行动作序列
  - 每步发布反馈 (step_index/step_count/phase/message)
```

### 5.2 执行前重新验证

在预览生成和实际执行之间，物体可能移动。执行前需要：

```python
def _revalidate(self, record):
    # 预览确认的 pick 位姿直接执行；执行前仍检查持物和安全状态。
    action["source"] = revalidate_candidate(original, "pick target", 0.002)

    # place 目标使用预览确认时的盒子位姿。
    action["destination"] = revalidate_candidate(original, "box", 0.05)
```

### 5.3 盒子重检测与重定位

放置阶段，盒子可能在 preview 和执行之间移动。系统采集 5 帧新鲜检测来处理：

```
采集 5 帧盒子检测样本
         │
         ▼
计算 5 帧 XY 位置的中位数
         │
         ▼
检查 5 帧间的最大两两 XY 距离 → < 1cm (stability_threshold)?
    NO ──→ 重新采集（盒子在移动中）
    YES
         │
         ▼
计算 preview 位置 vs 当前中位数的 XY 偏移
         │
    < 1cm (retarget_threshold)  → "unchanged" 用原位置
    < 5cm (max_shift)           → "relocate"  重定位到新位置
    ≥ 5cm                        → "reject"   拒绝执行，要求重新预览
```

预览确认后不进行盒子的多帧重检；执行前应保持盒子静止。

---

## 第六阶段：安全与状态管理

### 6.1 状态机

```
                    ┌─────────────────────────────────────┐
                    │              IDLE                    │
                    └──────────┬──────────────────────────┘
                               │ preview
                               ▼
                    ┌──────────────────────┐
                    │   PREVIEW_READY      │
                    │   (15秒有效期)        │
                    └──────────┬───────────┘
                               │ execute (y)
                               ▼
                    ┌──────────────────────┐
                    │     EXECUTING        │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
        pick only        place 失败          全部完成
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────┐    ┌────────────────┐    ┌──────────┐
    │  HOLDING  │                    │IDLE/HOLDING│
    │(持有物体) │    │(等待放置重试)  │    └──────────┘
    └─────┬─────┘    └───────┬────────┘
          │                  │
          │ 新 preview
          │ (pick_place)     │ → execute
          │                  │
          ▼                  ▼
    ┌──────────────┐    ┌──────────────┐
    │  EXECUTING   │    │  EXECUTING   │
    └──────────────┘    └──────────────┘
```

**任意状态 → STOPPED / RESETTING / RESET_FAILED**：由用户按键触发

### 6.2 Safety Epoch 机制

```python
# 每次 stop/reset 递增 epoch
# 所有预览和执行都绑定到特定 epoch
# epoch 变化 → 之前的所有预览立即失效

def safety_execution_valid(state, execution_epoch):
    return not state.blocked and state.epoch == execution_epoch
```

### 6.3 持有状态管理

```python
# self._held_source — 当前夹爪中持有的物体 (ResolvedCandidate)

# HOLDING 状态: 刚完成 pick，等待用户下一个指令
# 放置阶段失败后保持 HOLDING，操作者可创建新的放置预览。
```

### 6.4 键盘快捷键

| 按键 | 命令 | 效果 |
|------|------|------|
| 空格 | stop | 立即停止 + 取消当前 Action |
| h | reset | 停止 → 打开夹爪 → 回 Home |
| r | resume | 清除停止状态（不恢复已取消任务） |
| y | - | 确认执行当前预览 |
| n | - | 丢弃当前预览 |

---

## 完整端到端数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  相机硬件                                                           │
│  Realsense / OAK-D                                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ RGB Image + Depth Image
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  visual_perception / llm_visual_perception.py                            │
│  - YOLOv8 OBB 推理 (Ultralytics)                                    │
│  - 发布 /yolo/detected_result (class_name, confidence, 4角点像素坐标) │
│  - 发布 /yolo/detected_result/depth (对齐深度图)                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ ROS2 Topics
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LlmControlTaskServer._resolve_candidate()                         │
│  - YOLO + Depth 时间同步 (<50ms 容差)                                │
│  - robust_center3d_from_obb_depth (MAD 异常值剔除)                   │
│  - OBB 最长边 → yaw 角计算                                          │
│  - TF: camera_frame → base_link                                     │
│  - 输出: ResolvedCandidate (index, class, confidence, uv, xyz, yaw) │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ candidates JSON
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  用户输入: "抓取最左边的螺丝钉"                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LlmControlTaskServer._llm_plan()                                  │
│  - instruction_has_visual_intent (视觉意图检测)                       │
│  - 本地确定性目标选择（类别/图像方位/末端距离）                         │
│  - _has_disambiguator (消歧词检测)                                   │
│  - 构建 context JSON: {instruction, candidates, holding, pose}       │
│  - 多轮对话历史压缩（语义级，去除具体 index）                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ context JSON
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DeepSeekClient.chat()                                              │
│  POST https://api.deepseek.com/chat/completions                     │
│  - model: deepseek-chat                                             │
│  - response_format: {"type": "json_object"}                         │
│  - System Prompt + 历史 + 当前上下文                                 │
│  - 返回: {"actions": [{"type":"pick","source_index":0}]}             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ JSON response
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  parse_llm_plan() — 多层验证                                        │
│  1. JSON 合法性检查                                                  │
│  2. Schema 字段验证                                                  │
│  3. Candidate index 存在性验证                                       │
│  4. Class 约束验证 (source∈pick_classes, dest∈place_classes)         │
│  5. 消歧检查 (同类多物体 + 无消歧词 → 拒绝)                           │
│  6. Intent 一致性验证 (视觉意图↔视觉动作)                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ TaskPlan (verified)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  _enrich_plan() — 动作富化                                           │
│  - 重新 resolve 每个视觉目标的 3D 位姿                                │
│  - 计算 approach/grasp/carry/release 高度                           │
│  - 计算 grasp quaternion (roll=-180°, yaw=物体yaw+π/2)              │
│  - _check_pose: 工作空间 + 碰撞感知 IK 验证                          │
│  - 展开为 6-13 步具体执行序列                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ enriched plan + public preview JSON
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  返回给用户: Preview JSON                                            │
│  包含: preview_id, detections, steps, valid_for_sec, checks          │
│  用户看到 13 步详细计划，输入 y 确认                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ y (确认)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  _execute_preview() — 逐步执行                                       │
│  - _revalidate: 重新检测物体位置                                      │
│  - 执行 PICK 阶段 (open→home→approach→grasp→close→carry)            │
│  - _collect_box_samples: 5帧盒子重检测+稳定性+重定位                  │
│  - 执行 PLACE 阶段 (approach→release→open→retreat→home→close)       │
│  - 每步发布 ExecutePreview.Feedback                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ Joint space / Cartesian 运动目标
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  myrobot_planning_core + trajectory_retime_server                    │
│  - MoveItMotion.move_to_pose (planning_client="fairino")            │
│  - BiRRT* 路径规划                                                   │
│  - TOTG 时间最优重定时                                                │
│  - ros2_control → fairino_hardware → libfairino.so → 电机            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 关键文件索引

| 文件 | 职责 |
|------|------|
| [llm_control_task_server.py](../../llm_arm_control/llm_arm_control_nodes/llm_control_task_server.py) | 主任务服务器：视觉同步、LLM 推理编排、动作富化、执行 |
| [task_logic.py](../../llm_arm_control/llm_arm_control_nodes/task_logic.py) | 纯函数库：JSON 解析、计划验证、消歧、安全状态、盒子重定位 |
| [deepseek_client.py](../../llm_arm_control/llm_arm_control_nodes/deepseek_client.py) | DeepSeek API 客户端（纯标准库，无第三方依赖） |
| [deepseek_credentials.py](../../llm_arm_control/llm_arm_control_nodes/deepseek_credentials.py) | API Key 管理（GNOME Keyring） |
| [robot_pose_control_server.py](../../llm_arm_control/llm_arm_control_nodes/robot_pose_control_server.py) | 基类：MoveIt2 初始化、运动执行、夹爪控制 |
| [llm_control_cli.py](../../llm_arm_control/llm_arm_control_nodes/llm_control_cli.py) | 交互式终端客户端 |

---

## 设计哲学

1. **LLM 不做视觉** — LLM 只处理结构化数值，视觉理解完全交给 YOLO + 深度估计
2. **LLM 不生成坐标** — LLM 只用系统提供的 candidate index，禁止自创位姿
3. **多层防御验证** — LLM 输出经过 Schema → Candidate → Intent → IK 四层验证
4. **两阶段提交** — 预览-确认模式防止 LLM 幻觉导致误操作
5. **保守消歧策略** — 宁可拒绝执行，也不冒险猜测用户的模糊指令
6. **状态机驱动** — 所有操作都有明确的状态约束（持有、恢复、停止）
