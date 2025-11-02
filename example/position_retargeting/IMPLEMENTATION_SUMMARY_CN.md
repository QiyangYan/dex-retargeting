# Shadow Hand to OmniHand Retargeting 实现总结

## 概述

本实现参考了 `hand_robot_viewer.py` 中对 MANO 手的 retarget 处理方式，成功实现了从 Shadow Hand 到 OmniHand 的姿态重定向（retargeting）功能。核心思路是使用正向运动学（FK）从 Shadow Hand qpos 提取关节位置，然后使用优化器将这些位置映射到 OmniHand。

## 核心功能

### 1. FK 提取关节位置

**文件**: `dexonomy_viewer.py`

**方法**: `_compute_shadow_joint_positions(shadow_qpos)`

```python
def _compute_shadow_joint_positions(self, shadow_qpos: np.ndarray) -> np.ndarray:
    """
    从 Shadow Hand qpos 计算关节位置
    
    输入:
        - shadow_qpos: (29,) 数组
          [0:3]: base position (x, y, z)
          [3:7]: base quaternion (w, x, y, z)
          [7:29]: 22 joint angles
    
    输出:
        - joint_positions: (21, 3) 世界坐标系下的关节位置（匹配 MANO 手的 21 个关节）
    """
```

**实现步骤**:
1. 从 29D qpos 中提取 22 个关节角度
2. 使用 Pinocchio (`RobotWrapper`) 进行 FK 计算
3. 提取关键链接位置（指尖、中间关节等）
4. 使用 base pose (位置和四元数) 转换到世界坐标系

**提取的关节** (21个，匹配 MANO 手的关节顺序):
- 索引 0: `palm` (wrist)
- 索引 1-4: `thbase`, `thproximal`, `thmiddle`, `thtip` (thumb: mcp, pip, dip, tip)
- 索引 5-8: `ffknuckle`, `ffproximal`, `ffmiddle`, `fftip` (index: mcp, pip, dip, tip)
- 索引 9-12: `mfknuckle`, `mfproximal`, `mfmiddle`, `mftip` (middle: mcp, pip, dip, tip)
- 索引 13-16: `rfknuckle`, `rfproximal`, `rfmiddle`, `rftip` (ring: mcp, pip, dip, tip)
- 索引 17-20: `lfknuckle`, `lfproximal`, `lfmiddle`, `lftip` (little: mcp, pip, dip, tip)

### 2. Retargeting 到目标机器人

**方法**: `retarget_to_robot(shadow_qpos, robot_idx)`

**支持的优化类型**:
- **VECTOR**: 基于向量的优化（关节间的向量差）
- **FINGERTIP**: 基于指尖位置的优化
- **POSITION**: 基于关节位置的优化
- **DEXPILOT**: DexPilot 风格的优化

**实现流程**:
```
Shadow Hand qpos (29D)
    ↓
FK 计算
    ↓
关节位置 (21×3，匹配 MANO 手)
    ↓
根据 retargeting_type 准备参考值
    ↓
第一个优化器 (例如 VECTOR)
    ↓
初步 qpos
    ↓
[可选] 第二个优化器 (例如 FINGERTIP)
    ↓
最终 OmniHand qpos
```

### 3. 双优化器支持

**配置**: `two_optimizers=True`

**推荐组合**: VECTOR (第一个) + FINGERTIP (第二个)

**工作原理**:
1. **第一个优化器 (VECTOR)**:
   - 优化全局手腕姿态
   - 优化手指大致方向
   - 输出初步结果

2. **第二个优化器 (FINGERTIP)**:
   - 使用第一个优化器的结果作为 warm start
   - 精细调整手指位置以匹配指尖
   - 输出最终精确结果

**优点**:
- 更高的精度
- 更好的指尖匹配
- 适合离线数据处理

**缺点**:
- 计算时间增加（约 2 倍）
- 不适合实时应用

## 修改的文件

### 1. `dexonomy_viewer.py`

**新增导入**:
```python
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.robot_wrapper import RobotWrapper
```

**新增参数** (`__init__`):
- `hand_type`: HandType (默认 HandType.right)
- `retargeting_type`: RetargetingType (默认 RetargetingType.vector)
- `two_optimizers`: bool (默认 False)
- `second_optimizer_type`: str (默认 "FINGERTIP")

**新增成员变量**:
- `self.retargetings`: List[SeqRetargeting] - retargeting 对象列表
- `self.second_retargeting`: Optional[SeqRetargeting] - 第二个优化器
- `self.shadow_robot_wrapper`: RobotWrapper - Shadow Hand FK 包装器

**新增方法**:
- `_compute_shadow_joint_positions()`: 从 Shadow Hand qpos 计算关节位置
- 修改 `_load_robots()`: 支持加载多个机器人和 retargeting 配置
- 修改 `retarget_to_robot()`: 实现完整的 retargeting 逻辑
- 修改 `render_grasp_single()`: 调用 retargeting

### 2. `visualize_dexonomy_grasp.py`

**新增导入**:
```python
from dex_retargeting.constants import HandType, RetargetingType
```

**新增参数** (所有函数):
- `hand_type`: HandType
- `retargeting_type`: RetargetingType
- `two_optimizers`: bool
- `second_optimizer_type`: str

**更新文档**: 添加了使用双优化器的示例

### 3. 新增文件

#### `README_DEXONOMY_RETARGETING.md`
- 详细的使用指南
- 参数说明
- 技术实现细节
- 故障排查

#### `test_dexonomy_retargeting.sh`
- 自动化测试脚本
- 包含 3 个测试场景

#### `example_shadow_to_omni.py`
- 4 个完整的使用示例
- 单优化器示例
- 双优化器示例
- 批量处理示例
- 比较不同优化器示例

#### `IMPLEMENTATION_SUMMARY_CN.md` (本文件)
- 实现总结
- 中文文档

## 使用方法

### 基础用法

```bash
# Shadow Hand + OmniHand (单优化器 VECTOR)
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --data_idx 0
```

### 推荐用法（双优化器）

```bash
# Shadow Hand + OmniHand (VECTOR + FINGERTIP)
python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers True \
    --second_optimizer_type FINGERTIP \
    --data_idx 0
```

### 运行测试

```bash
# 运行自动化测试
./test_dexonomy_retargeting.sh

# 或运行示例脚本
python example_shadow_to_omni.py 2  # 运行示例 2（双优化器）
```

## 技术细节

### Shadow Hand qpos 格式

```
[0:3]   - base position (x, y, z)
[3:7]   - base quaternion (w, x, y, z)
[7:29]  - 22 joint angles
```

### 关节位置提取

使用 Pinocchio 的 FK：
```python
self.shadow_robot_wrapper.compute_forward_kinematics(qpos_for_fk)
link_pose = self.shadow_robot_wrapper.get_link_pose(link_id)
position = link_pose[:3, 3]
```

### 坐标系转换

从局部坐标系转换到世界坐标系：
```python
# 提取 base pose
base_pos = shadow_qpos[0:3]
base_quat_wxyz = shadow_qpos[3:7]

# 转换为旋转矩阵
r = Rotation.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], 
                       base_quat_wxyz[3], base_quat_wxyz[0]])
rot_matrix = r.as_matrix()

# 应用变换
joint_positions_world = (rot_matrix @ joint_positions.T).T + base_pos
```

### Retargeting 优化

根据 retargeting 类型准备参考值：

```python
# VECTOR 类型
if retargeting_type == "VECTOR":
    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    ref_value = joint_positions[task_indices, :] - joint_positions[origin_indices, :]

# FINGERTIP 类型
elif retargeting_type == "FINGERTIP":
    ref_value = joint_positions[indices, :]

# 优化
qpos_full = retargeting.retarget(ref_value)
```

### 双优化器实现

```python
# 第一个优化器
qpos_full = retargeting.retarget(ref_value)

# 第二个优化器
if self.two_optimizers:
    # 使用第一个优化器的结果作为 warm start
    self.second_retargeting.last_qpos = qpos_full[
        self.second_retargeting.optimizer.idx_pin2target
    ]
    
    # 执行第二次优化
    qpos_second = self.second_retargeting.retarget(second_ref_value)
    qpos = qpos_second  # 使用第二个优化器的结果
```

## 参考实现

本实现参考了 `hand_robot_viewer.py` 中的以下部分：

1. **MANO 手的 FK 计算**: `_compute_hand_geometry()` 方法
2. **双优化器配置**: `__init__()` 中的 `two_optimizers` 参数
3. **Retargeting 流程**: `render_dexycb_data()` 中的优化逻辑
4. **参考值准备**: 根据 retargeting_type 选择不同的参考值格式

## 性能考虑

### 单优化器 (VECTOR)
- **速度**: 快 (~50-100ms/帧)
- **精度**: 中等
- **适用**: 实时应用、快速预览

### 双优化器 (VECTOR + FINGERTIP)
- **速度**: 较慢 (~100-200ms/帧)
- **精度**: 高
- **适用**: 离线处理、高质量数据生成

## 调试信息

代码中包含详细的调试输出：
```python
cprint(f"[DEBUG] Retargeting type: {retargeting_type}", "cyan")
cprint(f"[DEBUG] Joint positions shape: {joint_positions.shape}", "white")
cprint(f"[DEBUG] Indices shape: {indices.shape}", "white")
```

可以通过这些信息验证：
- Retargeting 类型是否正确
- 关节位置维度是否匹配
- 索引是否正确

## 已知问题和解决方案

### 问题 1: 链接名称不匹配
**症状**: `ValueError: Link xxx not found`

**解决**: 已更新 `_compute_shadow_joint_positions` 中的链接名称，使用正确的 Shadow Hand URDF 链接名称（不带 "rh_" 前缀）

### 问题 2: 关节位置维度不匹配
**症状**: 优化器索引超出范围

**解决**: 确保提取的关节位置数量与 retargeting 配置中的 `target_link_human_indices` 匹配

### 问题 3: 坐标系不一致
**症状**: Retarget 后的机器人位置不正确

**解决**: 正确处理 base pose 变换，将局部坐标系转换到世界坐标系

## 未来改进方向

1. **性能优化**:
   - 缓存 FK 计算结果
   - 使用 GPU 加速优化
   - 并行处理多个 grasps

2. **功能扩展**:
   - 支持更多机器人类型
   - 添加碰撞检测
   - 实现逆向 retargeting (OmniHand → Shadow Hand)

3. **质量提升**:
   - 添加关节角度平滑
   - 实现自适应优化权重
   - 支持部分关节固定

## 文件结构

```
retarget/dex-retargeting/example/position_retargeting/
├── dexonomy_viewer.py                    # 主 viewer 实现（已修改）
├── visualize_dexonomy_grasp.py          # 可视化脚本（已修改）
├── dexonomy_dataset.py                   # 数据集加载器（未修改）
├── README_DEXONOMY_RETARGETING.md       # 使用指南（新增）
├── IMPLEMENTATION_SUMMARY_CN.md          # 实现总结（本文件，新增）
├── test_dexonomy_retargeting.sh         # 测试脚本（新增）
└── example_shadow_to_omni.py            # 示例脚本（新增）
```

## 依赖

- SAPIEN: 物理模拟和可视化
- Pinocchio: 正向运动学计算
- NumPy: 数值计算
- SciPy: 旋转变换
- dex-retargeting: Retargeting 优化器

## 联系和贡献

如有问题或建议，请参考：
- 原始实现: `hand_robot_viewer.py`
- 数据集: Dexonomy GRASP dataset
- 文档: `README_DEXONOMY_RETARGETING.md`

---

**实现完成日期**: 2025-11-01
**版本**: 1.0
**作者**: Based on hand_robot_viewer.py implementation

