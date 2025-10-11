# 🚀 快速开始指南 - 接触检测与可视化

## 📦 安装依赖

```bash
conda activate retarget
pip install trimesh rtree "pyglet<2"
```

## 🎯 基本用法

### 1. 仅可视化（不保存）

```bash
python store_hand_object.py --robots omni --data-id 0 --visualize
```

### 2. 保存接触信息（不可视化）

```bash
python store_hand_object.py --robots omni --save-contact-info --no-visualize
```

### 3. 同时保存和可视化

```bash
python store_hand_object.py --robots omni --save-contact-info --data-id 0 --visualize
```

### 4. 使用专用工具可视化接触关节

```bash
# 交互式 3D 查看器
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0

# 保存为图片
python visualize_contact.py contact_info_mesh_based_20251010_223301.npy --data-id 0 --save-image contact.png
```

## 🎨 可视化说明

### store_hand_object.py 可视化
- 显示完整的手-物体交互动画
- 支持 retarget 到机器人手
- 可以看到整个抓取序列

### visualize_contact.py 可视化
- 专门显示接触检测结果
- 🔴 红色球体 = 接触关节
- 🟢 绿色点 = 非接触关节  
- ⚪ 灰色 mesh = 物体
- 彩色线条 = 手部骨架

## 📊 完整工作流程

```bash
# 步骤 1: 处理数据并保存接触信息（快速，无可视化）
python store_hand_object.py --robots omni --save-contact-info --no-visualize

# 步骤 2: 读取摘要
python read_contact_info.py contact_info_mesh_based_*.npy

# 步骤 3: 可视化有接触的数据
python visualize_contact.py contact_info_mesh_based_*.npy --data-id 0

# 步骤 4: 保存可视化图片
python visualize_contact.py contact_info_mesh_based_*.npy --data-id 0 --save-image data0_contact.png
```

## 📝 参数说明

### store_hand_object.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--robots` | 机器人类型 | None |
| `--data-id` | 处理特定数据 | None（全部） |
| `--save-contact-info` | 保存接触信息 | False |
| `--visualize` | 显示可视化 | True |
| `--no-visualize` | 关闭可视化 | - |

### visualize_contact.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `contact_file` | 接触信息文件路径 | 必填 |
| `--data-id` | 可视化的数据 ID | 0 |
| `--save-image` | 保存截图路径 | None |

## 🎯 常见场景

### 场景 1: 调试单个数据

```bash
# 打开可视化查看抓取过程
python store_hand_object.py --robots omni --data-id 0 --visualize

# 查看接触检测结果
python store_hand_object.py --robots omni --data-id 0 --save-contact-info --visualize

# 3D 可视化接触关节
python visualize_contact.py contact_info_*.npy --data-id 0
```

### 场景 2: 批量处理数据

```bash
# 快速处理所有数据（无可视化）
python store_hand_object.py --robots omni --save-contact-info --no-visualize

# 之后选择性可视化
python visualize_contact.py contact_info_*.npy --data-id 0 --save-image data0.png
python visualize_contact.py contact_info_*.npy --data-id 5 --save-image data5.png
```

### 场景 3: 生成演示图片

```bash
# 批量生成所有有接触数据的可视化图片
for i in 0 5 10 15; do
    python visualize_contact.py contact_info_*.npy --data-id $i --save-image "demo_${i}.png"
done
```

## 🔧 输出文件

### 接触信息文件
- **位置**: `Dexycb_dataset/contact_info_mesh_based_YYYYMMDD_HHMMSS.npy`
- **格式**: NumPy 字典
- **内容**: 每个数据的 21 维接触标签数组

### 可视化图片
- **格式**: PNG (1920x1080)
- **内容**: 3D 渲染的手-物体接触可视化

## 💡 提示

1. **首次运行**: 建议先用 `--data-id 0` 测试单个数据
2. **批量处理**: 使用 `--no-visualize` 加快处理速度
3. **查看结果**: 用 `visualize_contact.py` 生成高质量可视化
4. **保存图片**: 添加 `--save-image` 避免交互式窗口

## ❓ 常见问题

**Q: 为什么没有看到可视化窗口？**
A: 检查是否使用了 `--no-visualize` 或 `--save-contact-info` 时没有加 `--visualize`

**Q: 如何同时保存和可视化？**
A: 使用 `--save-contact-info --visualize`

**Q: visualize_contact.py 需要重新处理数据吗？**
A: 不需要，它直接读取已保存的 `.npy` 文件

**Q: 如何批量可视化多个数据？**
A: 使用 bash 循环：
```bash
for i in {0..9}; do
    python visualize_contact.py contact_info.npy --data-id $i --save-image "contact_${i}.png"
done
```

## 📚 详细文档

- **[README_CONTACT_DETECTION.md](./README_CONTACT_DETECTION.md)**: 完整使用指南
- **[VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md)**: 可视化详细说明
- **[MESH_BASED_CONTACT_DETECTION.md](./MESH_BASED_CONTACT_DETECTION.md)**: 技术实现细节

---

**快速命令速查**

```bash
# 🎯 单数据 + 可视化
python store_hand_object.py --robots omni --data-id 0 --visualize

# 💾 批量处理
python store_hand_object.py --robots omni --save-contact-info --no-visualize

# 🎨 3D 可视化
python visualize_contact.py contact_info.npy --data-id 0

# 📸 保存图片
python visualize_contact.py contact_info.npy --data-id 0 --save-image output.png
```

