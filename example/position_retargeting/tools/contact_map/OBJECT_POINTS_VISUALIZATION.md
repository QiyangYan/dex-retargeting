# 物体表面点可视化功能

## 更新日期
2025-11-06

## 概述

添加了在 SAPIEN 3D 可视化中显示物体表面点的功能。这使得你可以：
- 验证物体点云采样的质量
- 检查物体表面点的分布
- 同时查看手部 contact map 和物体表面点
- 调试 contact map 计算

## 新功能

### `visualize_object_surface_points` 函数

```python
def visualize_object_surface_points(
    viewer: DexonomyGraspSAPIENViewer,
    object_points: np.ndarray,
    object_normals: Optional[np.ndarray] = None,
    sphere_radius: float = 0.003,
    max_points: int = 5000,
    color: Optional[np.ndarray] = None,
)
```

**特性：**
- 使用彩色球体可视化物体表面点
- 自动采样（如果点数超过 `max_points`）
- 可自定义颜色和球体大小
- 支持法向量（用于未来扩展）

## 使用方法

### 1. 仅显示物体表面点

```bash
cd /path/to/retarget/dex-retargeting/example/position_retargeting

python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0 \
    --show_object_points
```

**效果：**
- 物体表面显示浅蓝色球体
- 每个球体代表一个采样点
- 点分布显示物体表面几何

### 2. 同时显示 Contact Map 和物体表面点

```bash
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True
```

**效果：**
- **手部**：根据 contact map 值显示颜色渐变（蓝色→绿色→红色）
- **物体**：显示浅蓝色表面点
- 可以直观看到手与物体的接触关系

### 3. 自定义显示参数

可以在代码中调整：

```python
visualize_object_surface_points(
    viewer,
    object_points,
    sphere_radius=0.005,      # 更大的球体
    max_points=10000,          # 显示更多点
    color=np.array([1.0, 0.5, 0.0, 0.8])  # 橙色
)
```

## 参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--show_object_points` | bool | False | 是否显示物体表面点 |
| `--show_contact_map` | bool | False | 是否显示手部 contact map |
| `--contact_map_path` | str | None | Contact map 数据文件路径 |
| `--contact_map_idx` | int | None | 数据文件中的索引 |

### 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object_points` | ndarray (M, 3) | - | 物体表面点世界坐标 |
| `object_normals` | ndarray (M, 3) | None | 物体表面法向量（可选） |
| `sphere_radius` | float | 0.003 | 球体半径（米） |
| `max_points` | int | 5000 | 最大显示点数 |
| `color` | ndarray (4,) | [0.3, 0.6, 1.0, 0.8] | RGBA 颜色（浅蓝色） |

## 颜色方案

### 手部 Contact Map（当 `--show_contact_map True`）
- **蓝色** (contact value ≈ 0): 无接触
- **绿色** (contact value ≈ 0.5): 弱接触
- **红色** (contact value ≈ 1.0): 强接触

### 物体表面点（当 `--show_object_points True`）
- **浅蓝色** (默认): 所有物体表面点
- 与手部颜色区分明显

## 实际应用场景

### 场景 1: 验证点云采样
```bash
# 检查物体表面点分布是否均匀
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path data.npy \
    --contact_map_idx 0 \
    --show_object_points True
```

**检查要点：**
- 点分布是否均匀覆盖物体表面
- 是否有遗漏的区域
- 采样密度是否合适

### 场景 2: 调试 Contact Map
```bash
# 同时查看手部接触和物体表面
python tools/contact_map/visualize_dexonomy_grasp.py \
    --data_idx 0 \
    --contact_map_path data.npy \
    --contact_map_idx 0 \
    --show_contact_map True \
    --show_object_points True
```

**检查要点：**
- 红色（高接触）区域是否与物体表面对应
- 手部和物体之间的空间关系
- Contact map 计算是否合理

### 场景 3: 比较不同采样方法
```bash
# HandModel 方法
python store_dexonomy_retarget.py \
    --use_handmodel_method True \
    --hand_surface_sample_num 2000 \
    --max_objects 1

# SAPIEN 方法
python store_dexonomy_retarget.py \
    --use_handmodel_method False \
    --hand_surface_sample_num 2000 \
    --max_objects 1

# 可视化比较两种方法的结果
python tools/contact_map/visualize_dexonomy_grasp.py \
    --contact_map_path grasp_poses_retargeted_handmodel.npy \
    --contact_map_idx 0 \
    --show_object_points True
```

## 技术细节

### 数据来源

物体表面点从存储的数据中读取：

```python
contact_map_data = {
    "object_point_cloud": np.ndarray,      # (M, 3) 物体点
    "object_normal_cloud": np.ndarray,     # (M, 3) 物体法向量
    "hand_surface_points": np.ndarray,     # (N, 3) 手部点
    "hand_surface_normals": np.ndarray,    # (N, 3) 手部法向量
    "contact_map": np.ndarray,             # (N,) 接触值
}
```

### 渲染实现

使用 SAPIEN 的 `create_actor_builder()` API：

```python
# 为每个点创建球体 actor
for pos in object_points:
    builder = viewer.scene.create_actor_builder()
    builder.add_sphere_visual(radius=sphere_radius, material=material)
    marker = builder.build_static(name=f"object_point_marker_{i}")
    marker.set_pose(sapien.Pose(pos))
```

### 性能优化

- **自动采样**: 如果点数 > `max_points`，随机采样
- **静态 Actor**: 使用 `build_static()` 而非动态物体
- **批量清理**: 通过 `viewer.object_point_markers` 列表管理

## 示例输出

```
[INFO] Creating 2048 object surface point markers...
[INFO] Created 2048 object surface point markers
[INFO] Object surface points visualization added to scene
[INFO] Contact map visualization added to scene
```

## 故障排除

### 问题 1: 没有物体点显示

**症状：**
```
[WARNING] No object point cloud in contact map data
```

**原因：** 数据文件中缺少 `object_point_cloud`

**解决：**
- 确保使用最新版本的 `store_dexonomy_retarget.py`
- 重新生成数据文件

### 问题 2: 点显示太稀疏

**原因：** 默认 `max_points=5000` 可能对大物体不够

**解决：** 在代码中增加 `max_points`
```python
visualize_object_surface_points(
    viewer,
    object_points,
    max_points=10000  # 显示更多点
)
```

### 问题 3: 球体太小/太大

**解决：** 调整 `sphere_radius`
```python
visualize_object_surface_points(
    viewer,
    object_points,
    sphere_radius=0.005  # 更大的球体
)
```

## 未来扩展

### 可能的改进

1. **法向量可视化**
   - 显示小箭头表示法向量方向
   - 验证法向量是否正确

2. **颜色编码**
   - 根据法向量方向改变颜色
   - 根据距离手部的远近改变颜色

3. **交互式选择**
   - 点击球体查看详细信息
   - 高亮特定区域的点

4. **性能优化**
   - 使用 instanced rendering
   - LOD (Level of Detail) 系统

## 相关文件

- `visualize_dexonomy_grasp.py` - 主可视化脚本
- `store_dexonomy_retarget.py` - 数据生成脚本
- `SURFACE_POINT_UPDATE.md` - 表面点提取文档

## 参考

- SAPIEN Documentation: https://sapien.ucsd.edu/
- Contact Map 计算: `compute_cmap.py`
- 表面点提取: `get_hand_surface_points_from_handmodel()`

