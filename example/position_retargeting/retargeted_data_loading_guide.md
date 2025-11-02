# Retargeted Grasp Data 加载说明

本文档说明如何加载和使用 `grasp_poses_retargeted_*.npy` 文件中的数据。

## 数据结构

保存的 `.npy` 文件包含一个字典，其中每个键是数据索引（`data_idx`），对应一个抓取样本：

```python
{
    data_idx: {
        "object_id": str,           # 物体 ID
        "grasp_type": str,          # 抓取类型（如 "5_Light_Tool"）
        "scale_name": str,          # 缩放名称（如 "scale005"）
        "grasp_idx": int,           # 在原始文件中的抓取索引
        "scene_scale": float,       # 场景缩放因子（标量）
        "obj_scale": np.ndarray,    # 物体基础缩放 (3,) [sx, sy, sz]
        "shadow_qpos": np.ndarray,  # Shadow Hand 原始 qpos (29,)
        "robot_poses": {            # 各机器人的 retargeted qpos
            "RobotName.allegro": np.ndarray,  # 例如: Allegro Hand qpos
            "RobotName.omni_hand_right": np.ndarray,  # 例如: OmniHand qpos
            ...
        }
    },
    ...
}
```

## 加载示例

### 基本加载

```python
import numpy as np
from pathlib import Path

# 加载数据
data_path = Path("grasp_poses_retargeted_example.npy")
retargeted_data = np.load(data_path, allow_pickle=True).item()

# 访问第一个样本
sample_idx = 0
sample = retargeted_data[sample_idx]

print(f"Object ID: {sample['object_id']}")
print(f"Grasp Type: {sample['grasp_type']}")
print(f"Shadow Hand qpos shape: {sample['shadow_qpos'].shape}")
print(f"Available robots: {list(sample['robot_poses'].keys())}")
```

### 获取特定机器人的抓取姿态

```python
# 获取 Allegro Hand 的 retargeted qpos
robot_name = "RobotName.allegro"
if robot_name in sample['robot_poses']:
    allegro_qpos = sample['robot_poses'][robot_name]
    print(f"Allegro Hand qpos: {allegro_qpos}")
    print(f"DOF: {len(allegro_qpos)}")
```

## 物体缩放说明

### 缩放计算

在仿真器中加载物体时，**最终的物体缩放 = obj_scale × scene_scale**：

```python
# 获取缩放参数
obj_scale = sample['obj_scale']        # (3,) 例如: [0.05, 0.05, 0.05]
scene_scale = sample['scene_scale']    # 标量，例如: 1.2

# 计算最终缩放
final_scale = obj_scale * scene_scale  # (3,) 例如: [0.06, 0.06, 0.06]
```

### 在不同仿真器中应用

#### SAPIEN

```python
import sapien

# 创建场景
scene = engine.create_scene()

# 加载物体
builder = scene.create_actor_builder()
mesh_path = f"Dexonomy_dataset/objaverse_5k/processed_data/{sample['object_id']}/mesh/simplified.obj"

# 应用最终缩放
obj_scale = sample['obj_scale']
scene_scale = sample['scene_scale']
final_scale = obj_scale * scene_scale

builder.add_visual_from_file(
    filename=mesh_path,
    scale=final_scale  # 直接传入 (3,) 数组
)
obj = builder.build_static(name=sample['object_id'])
```

#### Isaac Sim / Isaac Gym

```python
# Isaac Sim 使用统一缩放或分量缩放
import omni.isaac.core

# 如果需要统一缩放，取平均值
obj_scale = sample['obj_scale']
scene_scale = sample['scene_scale']
uniform_scale = float(np.mean(obj_scale) * scene_scale)

# 或使用分量缩放
final_scale = obj_scale * scene_scale  # [sx, sy, sz]

# 在 Isaac Sim 中设置缩放
# prim.GetAttribute('xformOp:scale').Set((sx, sy, sz))
```

#### PyBullet

```python
import pybullet as p

# PyBullet 加载 URDF 时应用全局缩放
obj_scale = sample['obj_scale']
scene_scale = sample['scene_scale']

# PyBullet 只支持统一缩放
uniform_scale = float(np.mean(obj_scale) * scene_scale)

object_id = p.loadURDF(
    urdf_path,
    globalScaling=uniform_scale
)
```

## 手部姿态说明

### Shadow Hand qpos 格式

`shadow_qpos` 是 29 维数组，格式如下：

```
[0:3]   - 基座位置 (x, y, z)
[3:7]   - 基座四元数 (w, x, y, z)
[7:29]  - 22 个关节角度（Shadow Hand without wrist）
```

### Retargeted qpos 格式

每个机器人的 `qpos` 维度取决于机器人自由度：

- **Allegro Hand**: 16 DOF (手指关节)
- **OmniHand**: 28 DOF (6 DOF base + 22 DOF hand)
- **BarrettHand**: 8 DOF
- 等等...

**注意**：保存的 qpos **不包含 y_offset**，即机器人在原始位置。如需并排显示多个机器人，需自行添加偏移。

### 在仿真器中设置手部姿态

```python
# 示例：设置 Allegro Hand
allegro_qpos = sample['robot_poses']['RobotName.allegro']

# 在 SAPIEN 中设置
robot.set_qpos(allegro_qpos.astype(np.float32))

# 在 Isaac Sim 中设置
# 需要根据机器人 URDF 和关节映射来设置各关节
```

## 常见问题

### Q1: 为什么有两个缩放因子（obj_scale 和 scene_scale）？

- **obj_scale**: 物体的基础缩放，通常用于将物体归一化到合适的尺寸（如 0.05m）
- **scene_scale**: 场景特定的额外缩放因子，用于调整物体与手的相对尺寸以确保成功抓取

**最终缩放 = obj_scale × scene_scale** 才是物体在场景中的实际尺寸。

### Q2: 在其他仿真器中必须应用这两个缩放吗？

**是的！** 必须同时应用 `obj_scale` 和 `scene_scale`，否则：
- 只用 obj_scale：物体可能太小或太大，抓取失败
- 只用 scene_scale：物体尺寸错误
- 都不用：物体和手的相对尺寸完全错误

### Q3: 如何找到对应的物体 mesh？

```python
object_id = sample['object_id']
mesh_path = f"Dexonomy_dataset/objaverse_5k/processed_data/{object_id}/mesh/simplified.obj"
urdf_path = f"Dexonomy_dataset/objaverse_5k/processed_data/{object_id}/urdf/coacd.urdf"
```

### Q4: 如何批量处理所有样本？

```python
import numpy as np

# 加载所有数据
retargeted_data = np.load("grasp_poses_retargeted_example.npy", allow_pickle=True).item()

# 遍历所有样本
for data_idx, sample in retargeted_data.items():
    object_id = sample['object_id']
    grasp_type = sample['grasp_type']
    
    # 加载物体
    mesh_path = f"path/to/{object_id}/mesh/simplified.obj"
    final_scale = sample['obj_scale'] * sample['scene_scale']
    
    # 设置机器人
    for robot_name, robot_qpos in sample['robot_poses'].items():
        # 加载并设置机器人姿态
        pass
```

## 完整示例：在 SAPIEN 中加载

```python
import numpy as np
import sapien
from pathlib import Path

# 初始化引擎和场景
engine = sapien.Engine()
scene = engine.create_scene()

# 加载 retargeted data
data_path = Path("grasp_poses_retargeted_example.npy")
retargeted_data = np.load(data_path, allow_pickle=True).item()

# 选择一个样本
sample_idx = 0
sample = retargeted_data[sample_idx]

# 1. 加载物体
object_id = sample['object_id']
mesh_path = f"Dexonomy_dataset/objaverse_5k/processed_data/{object_id}/mesh/simplified.obj"

builder = scene.create_actor_builder()
final_scale = sample['obj_scale'] * sample['scene_scale']
builder.add_visual_from_file(filename=mesh_path, scale=final_scale)
obj = builder.build_static(name=object_id)

# 2. 加载机器人（假设是 Allegro Hand）
robot_name = "RobotName.allegro"
allegro_urdf = "path/to/allegro_hand.urdf"

loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(allegro_urdf)

# 3. 设置机器人姿态
if robot_name in sample['robot_poses']:
    robot_qpos = sample['robot_poses'][robot_name]
    robot.set_qpos(robot_qpos.astype(np.float32))

# 4. 渲染
scene.update_render()
print(f"Successfully loaded grasp for object {object_id}")
```

## 数据集路径

默认 Dexonomy 数据集结构：
```
Dexonomy_dataset/
├── objaverse_5k/
│   ├── processed_data/
│   │   ├── {object_id}/
│   │   │   ├── mesh/
│   │   │   │   └── simplified.obj
│   │   │   ├── urdf/
│   │   │   │   └── coacd.urdf
│   │   │   └── info/
│   │   │       └── simplified.json
│   ├── scene_cfg/
│   └── valid_split/
└── Dexonomy_GRASP_shadow/
    └── succ_collect/
```

## 总结

1. **加载数据**：使用 `np.load(..., allow_pickle=True).item()`
2. **物体缩放**：`final_scale = obj_scale * scene_scale`（两者都必须应用）
3. **手部姿态**：直接使用 `robot_poses[robot_name]` 中的 qpos
4. **物体路径**：根据 `object_id` 在 `objaverse_5k/processed_data/` 中查找

如有其他问题，请参考源代码：
- `store_dexonomy_retarget.py`: 数据生成
- `dexonomy_viewer.py`: SAPIEN 可视化示例

