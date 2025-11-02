# Shadow Hand to OmniHand Retargeting - 快速开始

## 一句话总结

使用 FK 和优化器，从 Shadow Hand 数据集 retarget 到 OmniHand，支持 VECTOR 和 FINGERTIP 两种优化方式。

## 最简单的使用方式

```bash
cd /home/guizhewei/guizhewei/retarget/dex-retargeting/example/position_retargeting

# 推荐方式：双优化器（VECTOR + FINGERTIP）
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --data_idx 0
```

## 三种主要配置

### 1. 快速模式（单优化器 VECTOR）
```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --data_idx 0
```
- ⚡ 速度快
- ✓ 适合实时应用
- ⚠️ 精度中等

### 2. 高质量模式（双优化器）
```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --data_idx 0
```
- ⭐ **推荐配置**
- 🎯 精度高
- ⚠️ 速度较慢

### 3. 指尖优化模式（单优化器 FINGERTIP）
```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type fingertip \
    --data_idx 0
```
- 🎯 专注指尖匹配
- ✓ 适合精细操作

## 运行示例

### 方式 1: 使用示例脚本
```bash
# 运行推荐配置（示例 2）
python example_shadow_to_omni.py 2

# 或选择其他示例
python example_shadow_to_omni.py 1  # 单优化器
python example_shadow_to_omni.py 3  # 批量处理
python example_shadow_to_omni.py 4  # 比较不同优化器
```

### 方式 2: 运行自动测试
```bash
./test_dexonomy_retargeting.sh
```

## 核心实现

### 1. FK 提取关节位置
```python
# 在 dexonomy_viewer.py 中
joint_positions = self._compute_shadow_joint_positions(shadow_qpos)
```

### 2. Retarget 到目标机器人
```python
# 第一个优化器
qpos = retargeting.retarget(ref_value)

# [可选] 第二个优化器
if two_optimizers:
    qpos = second_retargeting.retarget(second_ref_value)
```

## 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--robots` | 机器人列表 | `shadow_no_wrist omni` |
| `--retargeting_type` | 第一个优化器类型 | `vector` |
| `--two_optimizers` | 是否使用双优化器 | `True` |
| `--second_optimizer_type` | 第二个优化器类型 | `FINGERTIP` |
| `--data_idx` | 数据集索引 | `0` |

## 文件说明

| 文件 | 说明 |
|------|------|
| `dexonomy_viewer.py` | ⭐ 核心实现（已修改） |
| `visualize_dexonomy_grasp.py` | ⭐ 主入口脚本（已修改） |
| `example_shadow_to_omni.py` | 📖 示例脚本 |
| `test_dexonomy_retargeting.sh` | 🧪 测试脚本 |
| `README_DEXONOMY_RETARGETING.md` | 📚 详细文档 |
| `IMPLEMENTATION_SUMMARY_CN.md` | 📚 实现总结 |

## 常见问题

### Q: 如何切换不同的 grasp？
```bash
# 使用 data_idx 参数
python visualize_dexonomy_grasp.py --robots shadow_no_wrist omni --data_idx 10
```

### Q: 如何只看特定类型的 grasp？
```bash
# 使用 grasp_type 参数
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --grasp_type "5_Light_Tool" \
    --data_idx 0
```

### Q: 如何保存视频？
```bash
# 使用 headless 模式
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --headless True
```

### Q: 双优化器相比单优化器有什么优势？
- ✅ 更精确的指尖位置匹配
- ✅ 更好的手指姿态
- ⚠️ 计算时间增加约 2 倍

### Q: 哪种配置最适合我？

| 场景 | 推荐配置 |
|------|----------|
| 实时应用 | 单优化器 VECTOR |
| 数据生成 | 双优化器 VECTOR + FINGERTIP |
| 快速预览 | 单优化器 VECTOR |
| 高质量结果 | 双优化器 VECTOR + FINGERTIP |

## 技术原理

```
Shadow Hand qpos (29D)
    ↓
FK (Pinocchio)
    ↓
关节位置 (21×3，匹配 MANO 手)
    ↓
[第一个优化器] VECTOR
    ↓
初步 OmniHand qpos
    ↓
[第二个优化器] FINGERTIP (可选)
    ↓
最终 OmniHand qpos
```

## 下一步

- 📖 阅读详细文档: [README_DEXONOMY_RETARGETING.md](README_DEXONOMY_RETARGETING.md)
- 📚 查看实现总结: [IMPLEMENTATION_SUMMARY_CN.md](IMPLEMENTATION_SUMMARY_CN.md)
- 🔍 探索源代码: `dexonomy_viewer.py`
- 🧪 运行测试: `./test_dexonomy_retargeting.sh`

## 参考

- 参考实现: `hand_robot_viewer.py` (MANO retargeting)
- 数据集: Dexonomy GRASP dataset
- 库: dex-retargeting, SAPIEN, Pinocchio

---

**版本**: 1.0 | **日期**: 2025-11-01

