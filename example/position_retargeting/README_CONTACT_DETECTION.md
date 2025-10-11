# DexYCB 接触检测 - 使用指南

## 🎯 功能概述

本项目实现了基于**真实 mesh 几何**的精确接触检测，用于判断 MANO 手部的 21 个关节是否与 YCB 物体接触。

## ✨ 核心特性

- ✅ 使用真实物体 mesh (`textured_simple.obj`)
- ✅ 精确计算关节到物体表面的距离（mm 级精度）
- ✅ 自动处理任意形状的物体
- ✅ 生成 21 维 0/1 接触标签数组
- ✅ 支持批量处理所有数据

## 🚀 快速开始

### 1. 安装依赖

```bash
conda activate retarget
pip install trimesh rtree
```

### 2. 运行接触检测

```bash
# 处理单个数据（推荐先测试）
python store_hand_object.py --robots omni --save-contact-info --data-id 0

# 处理所有数据
python store_hand_object.py --robots omni --save-contact-info
```

### 3. 查看结果

运行后会输出：
```
============================================================
Total contacts detected: 6/21
============================================================

Contact joints (6/21):
  [11] middle_dip      - distance: 0.0073m
  [12] middle_tip      - distance: 0.0093m
  [14] ring_pip        - distance: 0.0048m
  [15] ring_dip        - distance: 0.0051m
  [19] little_dip      - distance: 0.0089m
  [20] little_tip      - distance: 0.0038m

Contacts by finger:
  middle    : 2
  ring      : 2
  little    : 2
```

结果会自动保存到:
```
/path/to/Dexycb_dataset/contact_info_mesh_based_YYYYMMDD_HHMMSS.npy
```

## 📊 输出格式

### 保存的数据结构

```python
{
    0: {  # data_id
        "contact_labels": np.array([0,0,0,...,1,1]),  # (21,) - 主要数据
        "capture_name": "20200709_150654",
        "target_object_id": 13,
        "target_object_idx": 0,
        "last_frame": 71,
        "detection_method": "SAPIEN",
        "distances_to_center": np.array([...]),  # (21,) - 每个关节到表面的距离
        "mesh_radius": 0.0713,  # mesh 包围球半径
        "sphere_radius": 0.01,
        "distance_threshold": 0.01,
        "total_contacts": 6,
        "contacts_by_finger": {"thumb": 0, "index": 0, ...}
    },
    1: { ... },
    ...
}
```

### 读取数据

```python
import numpy as np

# 加载
data = np.load("contact_info_mesh_based_20251010_223301.npy", allow_pickle=True).item()

# 遍历所有数据
for data_id, contact_info in data.items():
    labels = contact_info['contact_labels']  # 21维数组
    num_contacts = labels.sum()
    print(f"Data {data_id}: {num_contacts} contacts")
    
    # 获取接触的关节索引
    contact_joints = np.where(labels == 1)[0]
    print(f"  Contact joints: {contact_joints}")
    
    # 获取距离信息
    distances = contact_info['distances_to_center']
    print(f"  Min distance: {distances.min()*1000:.2f}mm")
```

## 🔧 参数说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robots` | str | None | 机器人类型（omni/inspire/etc.） |
| `--save-contact-info` | flag | False | 是否保存接触信息 |
| `--data-id` | int | None | 处理特定数据（None=全部） |
| `--use-sapien-contact` | flag | True | 使用 mesh-based 方法 |

### 内部参数（代码中）

| 参数 | 值 | 说明 | 位置 |
|------|-----|------|------|
| `distance_threshold` | 0.01m | 接触距离阈值（1cm） | `store_hand_object.py:230` |

**调整阈值**:
```python
# 严格: 5mm - 只检测真实接触
distance_threshold = 0.005

# 推荐: 1cm - 考虑噪声和误差
distance_threshold = 0.01  # ← 当前使用

# 宽松: 2cm - 包含接近的情况
distance_threshold = 0.02
```

## 📈 测试结果

### Data 0 示例

```
物体: 024_bowl (碗)
Mesh: 8331 顶点, 15728 面

距离统计:
  最小距离: 3.75mm
  最大距离: 80.82mm
  平均距离: 19.62mm

接触情况（1cm 阈值）:
  ✓ Joint 11: middle_dip (7.3mm)
  ✓ Joint 12: middle_tip (9.3mm)
  ✓ Joint 14: ring_pip   (4.8mm)
  ✓ Joint 15: ring_dip   (5.1mm)
  ✓ Joint 19: little_dip (8.9mm)
  ✓ Joint 20: little_tip (3.8mm)

总计: 6/21 个关节接触
抓握类型: 三指抓握（中指+无名指+小指）
```

### 所有数据统计

```bash
python test_mesh_contact.py
```

输出示例:
```
Data 0: Min=3.75mm, Contacts@1cm: 6/21  ← 有接触
Data 1: Min=164mm, Contacts@1cm: 0/21   ← 无接触
Data 2: Min=322mm, Contacts@1cm: 0/21
...
```

## 🛠️ 工具脚本

### 1. test_mesh_contact.py

快速测试所有数据的接触距离统计

```bash
python test_mesh_contact.py
```

### 2. read_contact_info.py

读取和分析保存的接触信息

```bash
python read_contact_info.py contact_info_mesh_based_20251010_223301.npy
```

## 📖 详细文档

- **[MESH_BASED_CONTACT_DETECTION.md](./MESH_BASED_CONTACT_DETECTION.md)**: 实现原理和技术细节
- **[OBJECT_LOADING_EXPLANATION.md](./OBJECT_LOADING_EXPLANATION.md)**: 物体加载机制说明

## ⚙️ 实现原理

### 核心步骤

```
1. 加载物体 mesh
   ↓
   mesh = trimesh.load("models/024_bowl/textured_simple.obj")
   
2. 变换到相机坐标系
   ↓
   mesh.apply_transform(object_pose_from_labels)
   
3. 计算距离
   ↓
   distance = mesh.nearest.on_surface(joint_pos)
   
4. 判断接触
   ↓
   contact = (distance <= 0.01)
```

### 关键技术

- **Trimesh**: 加载和操作 mesh
- **R-tree**: 空间索引，加速最近邻查询
- **Labels 数据**: 提供相机坐标系下的关节位置和物体姿态

## ❓ 常见问题

### Q: 为什么有些数据检测不到接触？

**A**: DexYCB 序列的最后一帧不一定是抓握最紧的时刻。你可以：
1. 检查其他帧
2. 增大阈值（但会降低精度）
3. 使用 `test_mesh_contact.py` 查看距离统计

### Q: 1cm 阈值是否太大？

**A**: 考虑了以下因素：
- MANO 关节标注误差: ~5mm
- 深度相机噪声: ~5mm  
- 皮肤变形: ~2-3mm
- Mesh 简化误差: ~2mm

**推荐**: 5mm（严格）、1cm（推荐）、2cm（宽松）

### Q: 可以检测所有帧吗？

**A**: 可以！修改 `detect_contact_sapien()` 函数：

```python
# 原代码: 只检测最后一帧
last_label_file = label_files[-1]

# 改为: 遍历所有帧
for label_file in label_files:
    # ... 检测接触
```

### Q: 如何可视化接触？

**A**: 使用 trimesh 可视化：

```python
import trimesh

# 加载 mesh 和关节
mesh = trimesh.load("models/024_bowl/textured_simple.obj")
joints = trimesh.PointCloud(hand_joints_3d)

# 标记接触关节为红色
colors = np.array([[0,255,0] if label==0 else [255,0,0] 
                   for label in contact_labels])
joints.colors = colors

# 显示
scene = trimesh.Scene([mesh, joints])
scene.show()
```

## 🔍 调试技巧

### 1. 查看单个关节的距离

```python
from contact_detection_sapien import detect_contact_in_viewer

results = detect_contact_in_viewer(...)
for i, dist in enumerate(results['min_distances']):
    print(f"Joint {i}: {dist*1000:.2f}mm")
```

### 2. 检查 mesh 加载

```python
from contact_detection_sapien import load_object_mesh

mesh = load_object_mesh(13, Path("/path/to/models"))
print(f"Vertices: {len(mesh.vertices)}")
print(f"Bounds: {mesh.bounds}")
print(f"Center: {mesh.centroid}")
```

### 3. 验证坐标系

```python
# 检查 hand_joints_3d 和 object_pose 是否在同一坐标系
labels = np.load("labels_000071.npz")
print(f"Hand joints range: {labels['joint_3d'].min()}-{labels['joint_3d'].max()}")
print(f"Object position: {labels['pose_y'][0][:, 3]}")
# 应该在相似的数值范围内（都在相机坐标系）
```

## 📝 代码修改记录

### 主要修改

1. **contact_detection_sapien.py**
   - 添加 `load_object_mesh()`: 加载真实 mesh
   - 修改 `detect_contact_in_viewer()`: 使用 mesh 计算精确距离
   - 移除估计半径的代码

2. **store_hand_object.py**
   - 更新调用 `detect_contact_in_viewer()` 时传入 `object_id` 和 `models_dir`
   - 调整 `distance_threshold` 为 0.01m
   - 添加自动文件命名（时间戳）

3. **依赖**
   - 新增: `trimesh`, `rtree`
   - 保留: `numpy`, `scipy`, `sapien`

## 🎓 扩展应用

### 1. 抓握质量评估

```python
# 基于接触数量评估抓握
def evaluate_grasp_quality(contact_labels):
    thumb = contact_labels[1:5].sum()
    others = contact_labels[5:].sum()
    
    if thumb >= 1 and others >= 2:
        return "Good grasp"
    else:
        return "Weak grasp"
```

### 2. 接触热力图

```python
# 生成接触强度热力图
distances = contact_info['distances_to_center']
intensity = 1.0 / (distances + 0.001)  # 距离越小强度越大
```

### 3. 多物体接触

```python
# 检测与多个物体的接触
for obj_idx in range(len(ycb_ids)):
    contacts = detect_contact_in_viewer(..., object_idx=obj_idx)
    print(f"Object {obj_idx}: {contacts['contact_labels'].sum()} contacts")
```

## 📚 参考资料

- [DexYCB Dataset](https://dex-ycb.github.io/)
- [Trimesh Documentation](https://trimsh.org/)
- [MANO Hand Model](https://mano.is.tue.mpg.de/)
- [YCB Object Set](http://www.ycbbenchmarks.com/)

## 🙏 致谢

- DexYCB 团队提供高质量数据集
- Trimesh 提供强大的 mesh 处理工具
- SAPIEN 提供物理模拟框架

---

**最后更新**: 2025-10-10  
**作者**: Contact Detection Team  
**版本**: 2.0 (Mesh-based)

