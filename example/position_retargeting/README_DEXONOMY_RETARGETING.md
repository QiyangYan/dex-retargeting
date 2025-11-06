# Dexonomy Dataset Retargeting 使用指南

本指南介绍如何使用 FK 和优化器从 Shadow Hand 数据集 retarget 到 OmniHand。

## 功能概述

- **从 Shadow Hand FK 提取关节位置**：使用 Pinocchio 进行正向运动学计算，从 Shadow Hand qpos 提取关节位置
- **支持多种 Retargeting 类型**：
  - `VECTOR`：基于向量的优化（推荐用于全局姿态）
  - `FINGERTIP`：基于指尖位置的优化（推荐用于手指细节）
  - `POSITION`：基于关节位置的优化
  - `DEXPILOT`：DexPilot 风格的优化
- **双优化器支持**：可以使用两个优化器级联，第一个优化器优化全局姿态，第二个优化器精细化手指位置

## 使用示例

### 1. 基础使用：Shadow Hand + OmniHand（单优化器 VECTOR）

```bash
cd /home/guizhewei/guizhewei/retarget/dex-retargeting/example/position_retargeting

python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --data_idx 0
```

### 2. 双优化器：VECTOR (全局) + FINGERTIP (手指细节)

**推荐配置**：这种配置先用 VECTOR 优化全局姿态和手腕位置，然后用 FINGERTIP 优化器精细调整手指位置。

```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers \
    --second_optimizer_type FINGERTIP \
    --data_idx 0
```

### 3. 单优化器：FINGERTIP

```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type fingertip \
    --data_idx 0
```

### 4. 查看多个连续 Grasp

```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --data_idx 0 \
    --num_grasps 5
```

### 5. 特定 Grasp 类型

```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --grasp_type "5_Light_Tool" \
    --data_idx 0
```

### 6. 渲染视频（Headless 模式）

```bash
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --data_idx 0 \
    --headless True
```

## 参数说明

### 必需参数
- `--robots`: 机器人列表，第一个必须是 `shadow_no_wrist`，后续可以添加 `omni`, `allegro`, `svh` 等

### Retargeting 参数
- `--retargeting_type`: 第一个优化器类型
  - `vector` (推荐): 基于向量的优化，适合全局姿态
  - `fingertip`: 基于指尖位置的优化
  - `position`: 基于关节位置的优化
  - `dexpilot`: DexPilot 风格的优化

- `--two_optimizers`: 是否使用两个优化器（默认 False）
  - `True`: 启用双优化器
  - `False`: 仅使用第一个优化器

- `--second_optimizer_type`: 第二个优化器类型（仅在 `two_optimizers=True` 时使用）
  - `FINGERTIP` (推荐): 精细化手指位置
  - `VECTOR`: 向量优化
  - `POSITION`: 位置优化
  - `DEXPILOT`: DexPilot 优化

### 数据集参数
- `--dexonomy_dir`: Dexonomy 数据集根目录（默认：`/home/guizhewei/guizhewei/Dexonomy_dataset`）
- `--data_idx`: 数据集中的 grasp 索引（默认：0）
- `--num_grasps`: 连续渲染的 grasp 数量（默认：1）
- `--grasp_type`: 特定的 grasp 类型（可选，例如 `"5_Light_Tool"`）
- `--object_id`: 特定的物体 ID（可选）
- `--split`: 数据集划分（`train`, `test`, 或 `all`，默认：`train`）

### 可视化参数
- `--headless`: 无头模式，保存视频而不显示窗口（默认：False）
- `--fps`: 渲染帧率（默认：10）
- `--show_ground`: 显示地面（默认：False）
- `--show_table`: 显示桌子（默认：False）
- `--use_tabletop_pose`: 使用 tabletop_pose.json 中的物体姿态（默认：True）

