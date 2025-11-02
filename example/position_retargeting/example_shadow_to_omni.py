#!/usr/bin/env python3
"""
示例脚本：从 Shadow Hand retarget 到 OmniHand

这个脚本展示了如何：
1. 从 Dexonomy 数据集加载 Shadow Hand grasp
2. 使用 FK 提取 Shadow Hand 关节位置
3. 使用优化器将 Shadow Hand retarget 到 OmniHand
4. 可视化结果
"""

from pathlib import Path
from dexonomy_dataset import DexonomyGraspDataset
from dexonomy_viewer import DexonomyGraspSAPIENViewer
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig

# 兼容性设置
import numpy as np
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def example_1_single_optimizer_vector():
    """
    示例 1: 使用单个 VECTOR 优化器
    
    适用场景：
    - 快速 retargeting
    - 实时应用
    - 全局姿态匹配
    """
    print("\n" + "="*60)
    print("示例 1: 单个 VECTOR 优化器")
    print("="*60)
    
    # 设置路径
    dexonomy_dir = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # 加载数据集
    print("[INFO] 加载 Dexonomy 数据集...")
    dataset = DexonomyGraspDataset(
        data_root=dexonomy_dir,
        split="train",
    )
    
    # 获取第一个 grasp
    grasp_data = dataset[0]
    print(f"[INFO] 加载 grasp: {grasp_data['grasp_type']}, 物体: {grasp_data['object_id'][:20]}...")
    
    # 创建 viewer（Shadow Hand + OmniHand）
    print("[INFO] 创建 SAPIEN viewer...")
    viewer = DexonomyGraspSAPIENViewer(
        robot_names=[RobotName.shadow_no_wrist, RobotName.omni],
        headless=False,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.vector,
        two_optimizers=False,  # 单优化器
    )
    
    # 渲染
    print("[INFO] 开始渲染...")
    viewer.render_grasp_single(grasp_data, fps=10)
    print("[INFO] 完成！")


def example_2_dual_optimizers():
    """
    示例 2: 使用双优化器 (VECTOR + FINGERTIP)
    
    适用场景：
    - 高质量 retargeting
    - 需要精确匹配指尖位置
    - 离线数据处理
    """
    print("\n" + "="*60)
    print("示例 2: 双优化器 (VECTOR + FINGERTIP)")
    print("="*60)
    
    # 设置路径
    dexonomy_dir = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # 加载数据集
    print("[INFO] 加载 Dexonomy 数据集...")
    dataset = DexonomyGraspDataset(
        data_root=dexonomy_dir,
        split="train",
    )
    
    # 获取第一个 grasp
    grasp_data = dataset[0]
    print(f"[INFO] 加载 grasp: {grasp_data['grasp_type']}, 物体: {grasp_data['object_id'][:20]}...")
    
    # 创建 viewer（Shadow Hand + OmniHand）
    print("[INFO] 创建 SAPIEN viewer...")
    viewer = DexonomyGraspSAPIENViewer(
        robot_names=[RobotName.shadow_no_wrist, RobotName.omni],
        headless=False,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.vector,  # 第一个优化器
        two_optimizers=True,  # 启用双优化器
        second_optimizer_type="FINGERTIP",  # 第二个优化器
    )
    
    # 渲染
    print("[INFO] 开始渲染...")
    viewer.render_grasp_single(grasp_data, fps=10)
    print("[INFO] 完成！")


def example_3_multiple_grasps():
    """
    示例 3: 批量处理多个 grasps
    
    适用场景：
    - 数据集批量转换
    - 评估不同 retargeting 方法
    - 生成训练数据
    """
    print("\n" + "="*60)
    print("示例 3: 批量处理多个 grasps")
    print("="*60)
    
    # 设置路径
    dexonomy_dir = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # 加载数据集
    print("[INFO] 加载 Dexonomy 数据集...")
    dataset = DexonomyGraspDataset(
        data_root=dexonomy_dir,
        grasp_type="5_Light_Tool",  # 只加载特定类型
        split="train",
    )
    
    print(f"[INFO] 数据集包含 {len(dataset)} 个 grasps")
    
    # 创建 viewer
    print("[INFO] 创建 SAPIEN viewer...")
    viewer = DexonomyGraspSAPIENViewer(
        robot_names=[RobotName.shadow_no_wrist, RobotName.omni],
        headless=False,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.vector,
        two_optimizers=True,
        second_optimizer_type="FINGERTIP",
    )
    
    # 处理前 5 个 grasps
    num_grasps = min(5, len(dataset))
    print(f"[INFO] 处理前 {num_grasps} 个 grasps...")
    
    for i in range(num_grasps):
        grasp_data = dataset[i]
        print(f"\n[INFO] Grasp {i+1}/{num_grasps}: {grasp_data['grasp_type']}, {grasp_data['object_id'][:20]}...")
        viewer.render_grasp_single(grasp_data, fps=10)
    
    print("[INFO] 完成！")


def example_4_compare_optimizers():
    """
    示例 4: 比较不同优化器
    
    展示如何比较不同 retargeting 方法的效果
    """
    print("\n" + "="*60)
    print("示例 4: 比较不同优化器")
    print("="*60)
    
    # 设置路径
    dexonomy_dir = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
    robot_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # 加载数据集
    dataset = DexonomyGraspDataset(data_root=dexonomy_dir, split="train")
    grasp_data = dataset[0]
    
    # 测试不同配置
    configs = [
        ("VECTOR", RetargetingType.vector, False, None),
        ("FINGERTIP", RetargetingType.fingertip, False, None),
        ("VECTOR + FINGERTIP", RetargetingType.vector, True, "FINGERTIP"),
    ]
    
    for name, retarget_type, two_opt, second_type in configs:
        print(f"\n[INFO] 测试配置: {name}")
        print("-" * 60)
        
        viewer = DexonomyGraspSAPIENViewer(
            robot_names=[RobotName.shadow_no_wrist, RobotName.omni],
            headless=False,
            hand_type=HandType.right,
            retargeting_type=retarget_type,
            two_optimizers=two_opt,
            second_optimizer_type=second_type if second_type else "FINGERTIP",
        )
        
        print(f"[INFO] 渲染配置: {name}")
        viewer.render_grasp_single(grasp_data, fps=10)
    
    print("[INFO] 所有测试完成！")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("Shadow Hand to OmniHand Retargeting 示例")
    print("="*60)
    
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
    else:
        print("\n请选择示例:")
        print("  1 - 单个 VECTOR 优化器 (快速)")
        print("  2 - 双优化器 VECTOR + FINGERTIP (高质量)")
        print("  3 - 批量处理多个 grasps")
        print("  4 - 比较不同优化器")
        print("\n使用方法: python example_shadow_to_omni.py [1-4]")
        print("或者直接运行以使用示例 2 (推荐)\n")
        example_num = 2
    
    if example_num == 1:
        example_1_single_optimizer_vector()
    elif example_num == 2:
        example_2_dual_optimizers()
    elif example_num == 3:
        example_3_multiple_grasps()
    elif example_num == 4:
        example_4_compare_optimizers()
    else:
        print(f"[ERROR] 无效的示例编号: {example_num}")
        sys.exit(1)

