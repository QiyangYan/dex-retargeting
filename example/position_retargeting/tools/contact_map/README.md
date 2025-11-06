# Contact Map 可视化工具

## 概述

这个目录包含用于可视化和调试 contact map 的工具，支持：
- 手部表面点可视化（基于 contact map 值的颜色编码）
- 物体表面点可视化（浅蓝色球体）
- 同时显示两者以验证接触关系

## 主要文件

| 文件 | 说明 |
|------|------|
| `visualize_dexonomy_grasp.py` | 主可视化脚本 |
| `OBJECT_POINTS_VISUALIZATION.md` | 物体表面点可视化文档 |
| `README.md` | 本文件 |

## 快速开始

### 1. 生成数据（包含 contact map 和表面点）

```bash
cd /path/to/retarget/dex-retargeting/example/position_retargeting

python store_dexonomy_retarget.py \
    --dexonomy_dir /path/to/Dexonomy_dataset \
    --use_handmodel_method True \
    --hand_surface_sample_num 2000 \
    --max_objects 10
```

### 2. 可视化 Contact Map（手部）

```bash
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0 \
    --show_contact_map True
```

**效果：**
- 手部表面显示颜色渐变：蓝色（无接触）→ 绿色（弱接触）→ 红色（强接触）

### 3. 可视化物体表面点

```bash
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0 \
    --show_object_points
```

**效果：**
- 物体表面显示浅蓝色球体
- 验证物体点云采样质量

### 4. 同时显示两者（推荐）

```bash
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True
```

**效果：**
- 同时看到手部接触和物体表面
- 验证 contact map 计算是否合理
- 检查手与物体的空间关系

## 主要参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--data_idx` | int | 数据集索引 | 0 |
| `--contact_map_path` | str | Contact map 数据文件路径 | None |
| `--contact_map_idx` | int | 数据文件中的索引 | None |
| `--show_contact_map` | bool | 显示手部 contact map | False |
| `--show_object_points` | bool | 显示物体表面点 | False |

## 颜色方案

### 手部 Contact Map
- **蓝色** → 无接触（contact value ≈ 0）
- **绿色** → 弱接触（contact value ≈ 0.5）
- **红色** → 强接触（contact value ≈ 1.0）

### 物体表面点
- **浅蓝色** → 所有采样点（与手部颜色区分）

## 使用场景

### 场景 1: 验证 Contact Map 计算
```bash
# 查看接触是否合理
python tools/contact_map/visualize_dexonomy_grasp.py \
    --contact_map_path data.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True
```

**检查：**
- 红色区域是否与物体接触
- 手部和物体之间的距离
- Contact map 分布是否合理

### 场景 2: 调试表面点采样
```bash
# 检查物体点云质量
python tools/contact_map/visualize_dexonomy_grasp.py \
    --contact_map_path data.npy \
    --contact_map_idx 0 \
    --show_object_points True
```

**检查：**
- 点分布是否均匀
- 是否覆盖整个物体表面
- 采样密度是否合适

### 场景 3: 比较不同方法
```bash
# 方法 1: HandModel
python tools/contact_map/visualize_dexonomy_grasp.py \
    --contact_map_path data_handmodel.npy \
    --contact_map_idx 0 \
    --show_contact_map True

# 方法 2: SAPIEN
python tools/contact_map/visualize_dexonomy_grasp.py \
    --contact_map_path data_sapien.npy \
    --contact_map_idx 0 \
    --show_contact_map True
```

## 数据格式

Contact map 数据文件应包含：

```python
{
    "contact_map": np.ndarray,              # (N,) 接触值 [0, 1]
    "hand_surface_points": np.ndarray,      # (N, 3) 手部点
    "hand_surface_normals": np.ndarray,     # (N, 3) 手部法向量
    "object_point_cloud": np.ndarray,       # (M, 3) 物体点
    "object_normal_cloud": np.ndarray,      # (M, 3) 物体法向量
}
```

## 故障排除

### 没有显示任何点
```
[WARNING] No object point cloud in contact map data
```
**解决：** 使用最新版本 `store_dexonomy_retarget.py` 重新生成数据

### 点太稀疏/太密
**解决：** 调整 `--hand_surface_sample_num` 参数重新生成数据

### 颜色不明显
**解决：** 检查 contact map 值的范围，确保有高接触区域

## 完整示例

```bash
# 1. 生成数据（包含所有需要的字段）
python store_dexonomy_retarget.py \
    --dexonomy_dir /home/guizhewei/guizhewei/Dexonomy_dataset \
    --robots shadow_no_wrist omni \
    --use_handmodel_method True \
    --hand_surface_sample_num 2000 \
    --max_objects 5 \
    --split train

# 2. 可视化第一个样本（同时显示手部和物体）
python tools/contact_map/visualize_dexonomy_grasp.py \
    --dexonomy_dir /home/guizhewei/guizhewei/Dexonomy_dataset \
    --data_idx 0 \
    --contact_map_path /home/guizhewei/guizhewei/Dexonomy_dataset/grasp_poses_retargeted_test.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True

# 3. 浏览多个样本
python tools/contact_map/visualize_dexonomy_grasp.py \
    --dexonomy_dir /home/guizhewei/guizhewei/Dexonomy_dataset \
    --data_idx 0 \
    --contact_map_path /home/guizhewei/guizhewei/Dexonomy_dataset/grasp_poses_retargeted_test.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True \
    --num_grasps 5
```

## 相关文档

- `../../SURFACE_POINT_UPDATE.md` - 表面点提取方法
- `../../README_CONTACT_MAP.md` - Contact Map 总文档
- `OBJECT_POINTS_VISUALIZATION.md` - 物体点可视化详细说明

## 更新日期

2025-11-06 - 添加物体表面点可视化功能
