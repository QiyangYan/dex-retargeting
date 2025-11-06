# Contact Map 计算功能说明

## 概述

已修改 `store_dexonomy_retarget.py` 脚本，添加了基于 `compute_cmap.py` 的 contact map 计算功能。现在脚本会自动为每个样本计算 Shadow Hand 和物体之间的接触图，并将其作为 `contact_map` 键存储到数据集中。

## 修改内容

### 1. 新增依赖
- `torch` 和 `torch.nn.functional`: 用于加速 contact map 计算
- `trimesh`: 用于网格处理（可选，主要用于法线计算）

### 2. 新增函数

#### `get_hand_surface_points_from_sapien(viewer, robot_idx=0)`
从 SAPIEN robot 中提取手部表面点（mesh vertices）。

**参数：**
- `viewer`: DexonomyGraspSAPIENViewer 实例
- `robot_idx`: 机器人索引（0 表示 Shadow Hand）

**返回：**
- `hand_surface_points`: (N, 3) 手部表面点世界坐标

#### `get_object_point_cloud_and_normals(viewer, object_idx=0, num_samples=2048)`
从 SAPIEN 物体中提取点云和法线。

**参数：**
- `viewer`: DexonomyGraspSAPIENViewer 实例
- `object_idx`: 物体索引
- `num_samples`: 采样点数量

**返回：**
- `point_cloud`: (num_samples, 3) 物体点云世界坐标
- `normal_cloud`: (num_samples, 3) 物体法线世界坐标

#### `compute_contact_map(hand_surface_points, object_point_cloud, object_normal_cloud, contact_threshold=0.02, use_torch=True)`
基于 `compute_cmap.py` 的逻辑计算 contact map。

**参数：**
- `hand_surface_points`: (N_hand, 3) 手部表面点
- `object_point_cloud`: (N_obj, 3) 物体点云
- `object_normal_cloud`: (N_obj, 3) 物体法线
- `contact_threshold`: 接触阈值（默认 0.02m）
- `use_torch`: 是否使用 PyTorch（默认 True，更快）

**返回：**
- `contact_value`: (N_hand,) 每个手部表面点的接触值（0-1 范围）

### 3. Contact Map 计算原理

基于 `compute_cmap.py` 中的能量对齐距离（energy align dist）方法：

1. **对齐距离计算：**
   ```
   align = (hand_point - obj_point) · obj_normal / distance
   align_dist = distance * exp(2 * (1 - align))
   ```

2. **接触距离：**
   ```
   contact_dist = sqrt(min(align_dist))
   ```

3. **接触值：**
   ```
   contact_value = 1 - 2 * (sigmoid(10 * contact_dist) - 0.5)
   ```

接触值范围 [0, 1]：
- 接近 1 表示强接触
- 接近 0 表示无接触

### 4. 数据存储格式

修改后的 `retargeted_poses_dict` 结构：

```python
{
    continuous_idx: {
        "original_idx": int,           # 原始数据集索引
        "target_object_name": str,     # 物体 ID
        "grasp_type": str,             # 抓取类型
        "scale_name": str,             # 缩放名称
        "grasp_idx": int,              # 抓取索引
        "scene_scale": float,          # 场景缩放因子
        "obj_scale": np.ndarray,       # 物体基础缩放 (3,)
        "target_pose_world": list,     # 物体姿态
        "shadow_qpos": np.ndarray,     # Shadow Hand qpos (29,)
        "robot_pose": list,            # Retargeted qpos 列表
        "contact_map": np.ndarray,     # NEW! Contact map (N_hand,)
    }
}
```

## 目录结构

工具脚本已整理到 `tools/contact_map/` 目录：

```
position_retargeting/
├── store_dexonomy_retarget.py      # 主脚本（包含 contact map 计算）
├── tools/
│   └── contact_map/
│       ├── __init__.py
│       ├── README.md               # 工具使用说明
│       ├── test_contact_map.py     # 测试脚本
│       └── visualize_contact_map.py # 可视化脚本
└── README_CONTACT_MAP.md          # 本文档
```

## 使用方法

### 基本用法（处理数据）
```bash
python store_dexonomy_retarget.py \
    --dexonomy-dir /path/to/Dexonomy_dataset \
    --robots shadow_no_wrist omni \
    --retargeting-type VECTOR \
    --split train \
    --max-objects 1
```

### 可视化模式（查看 contact map 效果）
```bash
python store_dexonomy_retarget.py \
    --dexonomy-dir /path/to/Dexonomy_dataset \
    --robots shadow_no_wrist omni \
    --retargeting-type VECTOR \
    --split train \
    --max-objects 1 \
    --visualize
```

### 测试 contact map 计算功能
```bash
cd /path/to/dex-retargeting/example/position_retargeting
python tools/contact_map/test_contact_map.py --test-computation
```

### 验证存储的数据
```bash
python tools/contact_map/test_contact_map.py \
    --data-path /path/to/grasp_poses_retargeted_xxx.npy
```

### 可视化 contact map
```bash
# 可视化单个样本
python tools/contact_map/visualize_contact_map.py \
    --data-path /path/to/grasp_poses_retargeted_xxx.npy \
    --sample-idx 0

# 分析所有样本
python tools/contact_map/visualize_contact_map.py \
    --data-path /path/to/grasp_poses_retargeted_xxx.npy \
    --analyze-all \
    --max-samples 10
```

## 读取存储的数据

```python
import numpy as np

# 加载数据
data = np.load("grasp_poses_retargeted_xxx.npy", allow_pickle=True).item()

# 访问某个样本的 contact map
sample_idx = 0
contact_map = data[sample_idx]["contact_map"]

print(f"Contact map shape: {contact_map.shape}")
print(f"Mean contact value: {contact_map.mean():.4f}")
print(f"Max contact value: {contact_map.max():.4f}")
print(f"Number of strong contacts (>0.5): {(contact_map > 0.5).sum()}")
```

## 性能说明

- **PyTorch 加速：**默认使用 PyTorch 进行矩阵运算，显著提升计算速度
- **点云采样：**物体点云默认采样 2048 个点，可根据需要调整
- **手部表面点：**直接提取 mesh vertices，数量取决于 URDF 中的网格细节

## 注意事项

1. **内存使用：**计算 contact map 需要额外内存，特别是手部表面点较多时
2. **计算时间：**每个样本增加约 0.5-2 秒的计算时间（取决于点数）
3. **错误处理：**如果 contact map 计算失败，会存储空数组 `np.array([])`，不会中断整个处理流程

## 故障排除

### 问题：Contact map 全为零或空
**原因：**
- 手部或物体的 mesh 提取失败
- 物体未正确加载

**解决：**
- 检查 URDF 文件中是否包含 visual mesh
- 确保物体文件路径正确
- 使用 `--visualize` 模式检查场景加载情况

### 问题：计算速度慢
**原因：**
- 手部表面点过多
- 未安装 PyTorch 或未使用 GPU

**解决：**
- 考虑对手部表面点进行下采样
- 安装 PyTorch GPU 版本
- 设置 `use_torch=False` 使用 NumPy（但会更慢）

## Contact Map 可视化

### 图表可视化（2D）

使用 matplotlib 查看 contact map 的统计分布：

```bash
python tools/contact_map/visualize_contact_map.py \
    --data-path /path/to/grasp_poses_retargeted_xxx.npy \
    --sample-idx 0
```

### SAPIEN 3D 可视化（推荐）

在 3D 场景中直观查看手部表面的接触分布：

```bash
python tools/contact_map/visualize_dexonomy_grasp.py \
    --dexonomy_dir /path/to/Dexonomy_dataset \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0
```

**颜色映射：**
- 🔵 **蓝色** - 低接触值（0.0-0.5），远离物体
- 🟢 **绿色** - 中等接触值（~0.5）
- 🔴 **红色** - 高接触值（0.5-1.0），与物体接触

**注意：**
- `data_idx` 是 Dexonomy 数据集的索引
- `contact_map_idx` 是 retargeted .npy 文件的连续索引
- 两者可能不同，使用样本的 `original_idx` 字段查找对应关系

详细使用说明见 `tools/contact_map/QUICKSTART.md`。

## 相关文件

- `store_dexonomy_retarget.py`: 主脚本（已修改，包含 contact map 计算）
- `tools/contact_map/test_contact_map.py`: 测试脚本
- `tools/contact_map/visualize_contact_map.py`: 图表可视化脚本（2D）
- `tools/contact_map/visualize_dexonomy_grasp.py`: SAPIEN 3D 可视化脚本（**推荐**）
- `tools/contact_map/README.md`: 工具使用说明
- `tools/contact_map/QUICKSTART.md`: 快速开始指南
- `tools/contact_map/CHANGELOG.md`: 功能更新日志
- `compute_cmap.py`: 原始 contact map 计算参考（位于 `example/utils/`）
- `contact_detection_sapien.py`: SAPIEN 接触检测工具

## 更新日期

2025-11-06 - 添加 SAPIEN 3D 可视化功能

