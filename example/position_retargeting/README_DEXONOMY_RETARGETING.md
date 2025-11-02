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
    --two_optimizers True \
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

## 技术实现细节

### 1. Shadow Hand FK 关节位置提取

```python
def _compute_shadow_joint_positions(self, shadow_qpos: np.ndarray) -> np.ndarray:
    """
    从 Shadow Hand qpos 计算关节位置
    
    Args:
        shadow_qpos: (29,) Shadow Hand qpos
            - [0:3]: base position (x, y, z)
            - [3:7]: base quaternion (w, x, y, z)
            - [7:29]: 22 joint angles
            
    Returns:
        joint_positions: (21, 3) 世界坐标系下的关节位置（匹配 MANO 手的 21 个关节）
    """
```

该方法：
1. 提取 22 个关节角度
2. 使用 Pinocchio 进行 FK 计算
3. 提取 21 个链接位置（匹配 MANO 手的关节顺序）：
   - wrist (palm)
   - thumb: mcp (thbase), pip (thproximal), dip (thmiddle), tip (thtip)
   - index: mcp (ffknuckle), pip (ffproximal), dip (ffmiddle), tip (fftip)
   - middle: mcp (mfknuckle), pip (mfproximal), dip (mfmiddle), tip (mftip)
   - ring: mcp (rfknuckle), pip (rfproximal), dip (rfmiddle), tip (rftip)
   - little: mcp (lfknuckle), pip (lfproximal), dip (lfmiddle), tip (lftip)
4. 转换到世界坐标系

### 2. Retarget 到目标机器人

```python
def retarget_to_robot(self, shadow_qpos: np.ndarray, robot_idx: int):
    """
    使用 FK 和优化将 Shadow Hand 姿态 retarget 到其他机器人
    """
```

该方法：
1. 计算 Shadow Hand 关节位置（FK）
2. 根据 retargeting 类型准备参考值
3. 第一个优化器优化全局姿态
4. （可选）第二个优化器精细化手指位置
5. 设置目标机器人的关节角度

### 3. 双优化器工作流程

1. **第一个优化器（例如 VECTOR）**：
   - 输入：Shadow Hand 关节位置
   - 优化：全局手腕姿态和手指大致位置
   - 输出：初步的 OmniHand qpos

2. **第二个优化器（例如 FINGERTIP）**：
   - 输入：Shadow Hand 指尖位置 + 第一个优化器的结果作为 warm start
   - 优化：精细调整手指位置以匹配指尖
   - 输出：最终的 OmniHand qpos

## 提示和技巧

1. **推荐配置**：对于大多数情况，使用 `VECTOR + FINGERTIP` 双优化器配置可以获得最佳结果

2. **调试模式**：代码中包含调试信息输出，可以查看：
   - Retargeting 类型
   - 关节位置 shape
   - 优化器索引信息

3. **性能考虑**：
   - 双优化器会增加计算时间，但通常结果更好
   - 对于实时应用，可以考虑只使用单优化器

4. **数据集浏览**：使用 `--data_idx` 参数浏览不同的 grasp 样本

## 故障排查

### 问题：找不到 Shadow Hand 链接
**解决方案**：检查 `_compute_shadow_joint_positions` 中的链接名称是否与 URDF 匹配

### 问题：Retargeting 结果不理想
**解决方案**：
1. 尝试不同的 retargeting 类型
2. 使用双优化器配置
3. 检查 Shadow Hand qpos 是否正确

### 问题：关节角度超出限制
**解决方案**：优化器会自动处理关节限制，如果仍有问题，检查 retargeting 配置文件

## 参考

- 参考实现：`hand_robot_viewer.py` 中的 MANO retargeting
- 相关文件：
  - `dexonomy_viewer.py`: SAPIEN viewer 实现
  - `dexonomy_dataset.py`: 数据集加载器
  - `visualize_dexonomy_grasp.py`: 主入口脚本

