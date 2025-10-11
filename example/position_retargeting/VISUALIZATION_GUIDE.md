# 接触检测可视化指南

## 🎨 可视化选项

### 方法 1: 使用 --visualize 参数

```bash
# 同时保存接触信息并显示可视化窗口
python store_hand_object.py --robots omni --save-contact-info --data-id 0 --visualize

# 只可视化，不保存接触信息（默认行为）
python store_hand_object.py --robots omni --data-id 0 --visualize

# 保存接触信息，不显示可视化（快速处理）
python store_hand_object.py --robots omni --save-contact-info --data-id 0 --no-visualize
```

### 方法 2: 使用专用的可视化工具

使用 `visualize_contact.py` 可视化接触关节的 3D 视图：

```bash
# 基本用法
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy

# 指定要可视化的数据 ID
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0

# 保存截图而不是交互式显示
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0 --save-image contact_vis.png
```

## 🎯 可视化效果说明

### visualize_contact.py 显示内容

1. **物体 Mesh** (灰色)
   - 使用真实的 `textured_simple.obj`
   - 半透明显示，可以看到内部的手部

2. **接触关节** (红色球体)
   - 距离物体表面 ≤ 1cm 的关节
   - 用红色小球突出显示
   - 显示距离信息（单位：mm）

3. **非接触关节** (绿色点)
   - 距离物体表面 > 1cm 的关节
   - 用绿色点显示

4. **手部骨架** (连线)
   - 连接相邻关节的线
   - 红色：至少一端有接触
   - 绿色：两端都无接触

### 示例输出

```
============================================================
🎨 Visualizing Contact Detection Results
============================================================

Loading mesh for object ID 13 (024_bowl)...
Loaded mesh for 024_bowl:
  Vertices: 8331
  Faces: 15728
  Bounds: [[-0.070247 -0.060539 -0.022485]
           [ 0.067601  0.068252  0.016343]]
  Centroid: [-0.00081316  0.00359061 -0.00325419]
  Bounding sphere radius: 0.0713m

📊 Contact Summary:
  Total contacts: 6/21
  Non-contacts: 15/21

✅ Contact joints:
    [11] middle_dip   (7.30mm)
    [12] middle_tip   (9.30mm)
    [14] ring_pip     (4.89mm)
    [15] ring_dip     (5.10mm)
    [19] little_dip   (8.92mm)
    [20] little_tip   (3.75mm)

🎨 Visualization Legend:
  🔴 Red spheres = Contact joints
  🟢 Green points = Non-contact joints
  ⚪ Gray mesh = Object
  Lines = Hand skeleton

============================================================
Opening 3D viewer... (close window to continue)
============================================================
```

## 📸 可视化示例

### 交互式 3D 查看器

使用 Trimesh 的内置查看器，支持：
- 🖱️ 鼠标拖动旋转视角
- 🔍 滚轮缩放
- ⌨️ 键盘快捷键：
  - `z`: 重置视角
  - `a`: 自动旋转
  - `w`: 线框模式
  - `f`: 全屏
  - `q`: 退出

### 保存截图

```bash
# 保存高清截图（1920x1080）
python visualize_contact.py contact_info.npy --data-id 0 --save-image contact_data0.png

# 批量保存多个数据的截图
for i in {0..9}; do
    python visualize_contact.py contact_info.npy --data-id $i --save-image "contact_data${i}.png"
done
```

## 🔧 自定义可视化

### 在代码中使用

```python
from visualize_contact import visualize_contact_3d
import numpy as np

# 准备数据
hand_joints_3d = np.load("hand_joints.npy")  # (21, 3)
contact_labels = np.array([0,0,0,1,1,...])   # (21,)
object_pose = np.array([x,y,z,qx,qy,qz,qw])  # (7,)
object_id = 13  # 024_bowl

# 可视化
visualize_contact_3d(
    hand_joints_3d=hand_joints_3d,
    contact_labels=contact_labels,
    object_pose=object_pose,
    object_id=object_id,
    models_dir=Path("/path/to/models"),
    distances=distances,  # Optional
    save_image="my_visualization.png"  # Optional
)
```

### 修改颜色

编辑 `visualize_contact.py` 中的颜色设置：

```python
# 接触关节颜色（默认：红色）
contact_cloud = trimesh.PointCloud(
    contact_joints,
    colors=[255, 0, 0, 255]  # [R, G, B, Alpha]
)

# 非接触关节颜色（默认：绿色）
non_contact_cloud = trimesh.PointCloud(
    non_contact_joints,
    colors=[0, 255, 0, 255]
)

# 物体颜色（默认：灰色半透明）
mesh_transformed.visual.face_colors = [200, 200, 200, 100]
```

### 调整球体大小

```python
# 接触关节球体半径（默认：5mm）
sphere = trimesh.primitives.Sphere(
    radius=0.005,  # 修改这里调整大小
    center=pos
)
```

## 📊 完整工作流程

### 1. 检测并保存接触信息

```bash
# 不显示可视化，快速处理
python store_hand_object.py --robots omni --save-contact-info --no-visualize
```

### 2. 查看保存的数据

```bash
# 读取数据摘要
python read_contact_info.py contact_info_mesh_based_20251010_223301.npy
```

### 3. 可视化特定数据

```bash
# 交互式查看
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0

# 或保存为图片
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0 --save-image data0.png
```

### 4. 批量处理和可视化

```bash
# 步骤 1: 处理所有数据（无可视化）
python store_hand_object.py --robots omni --save-contact-info --no-visualize

# 步骤 2: 找出有接触的数据
python -c "
import numpy as np
data = np.load('contact_info_mesh_based_*.npy', allow_pickle=True).item()
for data_id, info in data.items():
    if info['contact_labels'].sum() > 0:
        print(f'Data {data_id}: {info[\"contact_labels\"].sum()} contacts')
"

# 步骤 3: 可视化有接触的数据
python visualize_contact.py contact_info.npy --data-id 0 --save-image contact_0.png
```

## 🎬 视频录制

### 使用 ffmpeg 录制旋转视频

虽然当前工具不直接支持视频导出，但你可以：

1. **手动录制**：
   - 使用 OBS Studio 或其他屏幕录制软件
   - 在可视化窗口中手动旋转物体
   - 录制为视频

2. **批量截图合成**：
```python
# 在 visualize_contact.py 中添加多角度截图
for angle in range(0, 360, 30):
    scene.camera.look_at(...)
    scene.save_image(f"frame_{angle:03d}.png")

# 使用 ffmpeg 合成视频
# ffmpeg -framerate 10 -i frame_%03d.png -c:v libx264 contact_video.mp4
```

## 🐛 故障排除

### 问题 1: 窗口不显示

**原因**: 可能是 headless 环境或缺少显示支持

**解决**:
```bash
# 检查是否有 DISPLAY
echo $DISPLAY

# 如果没有，使用保存图片模式
python visualize_contact.py contact_info.npy --save-image output.png
```

### 问题 2: 颜色不正确

**原因**: 某些系统上 alpha 通道处理不同

**解决**: 修改 `visualize_contact.py` 中的颜色，使用完全不透明：
```python
colors=[255, 0, 0, 255]  # 确保 alpha=255
```

### 问题 3: Mesh 不显示

**原因**: Mesh 文件路径错误或加载失败

**解决**:
```bash
# 检查 mesh 文件是否存在
ls -lh /home/guizhewei/guizhewei/Dexycb_dataset/models/024_bowl/textured_simple.obj

# 检查加载日志
python visualize_contact.py contact_info.npy 2>&1 | grep -i "mesh\|error"
```

### 问题 4: 点太小看不清

**解决**: 增大球体半径
```python
# 在 visualize_contact.py 中修改
sphere = trimesh.primitives.Sphere(radius=0.01, center=pos)  # 从 0.005 改为 0.01
```

## 📚 参考命令速查

```bash
# 🎯 基础可视化
python store_hand_object.py --robots omni --data-id 0 --visualize

# 💾 保存接触信息 + 显示可视化
python store_hand_object.py --robots omni --save-contact-info --data-id 0 --visualize

# 🚀 快速处理（无可视化）
python store_hand_object.py --robots omni --save-contact-info --no-visualize

# 🎨 3D 可视化工具
python visualize_contact.py contact_info.npy --data-id 0

# 📸 保存截图
python visualize_contact.py contact_info.npy --data-id 0 --save-image output.png

# 📖 读取数据摘要
python read_contact_info.py contact_info.npy

# 🧪 测试距离统计
python test_mesh_contact.py
```

## 🎓 高级用法

### 叠加多个数据帧

```python
# 自定义脚本：可视化整个序列
import trimesh
import numpy as np

scene = trimesh.Scene()

# 添加物体
mesh = load_object_mesh(13, models_dir)
scene.add_geometry(mesh)

# 添加多帧手部关节（不同颜色）
for frame_idx in range(0, len(label_files), 10):
    joints = load_joints_for_frame(frame_idx)
    color_intensity = int(255 * frame_idx / len(label_files))
    cloud = trimesh.PointCloud(joints, colors=[color_intensity, 0, 255-color_intensity, 128])
    scene.add_geometry(cloud)

scene.show()
```

### 导出为其他格式

```python
# 导出为 PLY（可在 MeshLab 中打开）
scene.export('contact_visualization.ply')

# 导出为 OBJ
scene.export('contact_visualization.obj')

# 导出为 GLTF（可在浏览器中查看）
scene.export('contact_visualization.gltf')
```

---

**提示**: 所有可视化工具都支持 `--help` 参数查看完整选项！

```bash
python visualize_contact.py --help
python store_hand_object.py --help
```

