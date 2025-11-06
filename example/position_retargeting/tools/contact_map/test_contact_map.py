"""
测试 contact map 计算功能的简单脚本
"""
import sys
from pathlib import Path

# 添加父目录到路径，以便导入 store_dexonomy_retarget
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

import numpy as np
from termcolor import cprint

# 测试加载和验证存储的 contact map 数据
def test_load_retargeted_data(data_path: str):
    """
    测试加载带有 contact map 的 retargeted 数据
    
    Args:
        data_path: .npy 文件路径
    """
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"测试加载数据文件: {data_path}", "cyan")
    cprint(f"{'='*60}\n", "cyan")
    
    try:
        # 加载数据
        data = np.load(data_path, allow_pickle=True).item()
        cprint(f"✓ 成功加载数据，共 {len(data)} 个样本", "green")
        
        # 统计信息
        contact_map_stats = {
            "total_samples": len(data),
            "with_contact_map": 0,
            "empty_contact_map": 0,
            "contact_map_shapes": [],
            "mean_contact_values": [],
            "max_contact_values": [],
        }
        
        # 遍历所有样本
        for idx, sample in data.items():
            if "contact_map" in sample:
                contact_map = sample["contact_map"]
                
                if len(contact_map) > 0:
                    contact_map_stats["with_contact_map"] += 1
                    contact_map_stats["contact_map_shapes"].append(contact_map.shape)
                    contact_map_stats["mean_contact_values"].append(contact_map.mean())
                    contact_map_stats["max_contact_values"].append(contact_map.max())
                else:
                    contact_map_stats["empty_contact_map"] += 1
        
        # 打印统计信息
        cprint(f"\n{'='*60}", "white")
        cprint(f"Contact Map 统计信息", "white")
        cprint(f"{'='*60}", "white")
        cprint(f"总样本数: {contact_map_stats['total_samples']}", "white")
        cprint(f"包含 contact map: {contact_map_stats['with_contact_map']}", "green")
        cprint(f"空 contact map: {contact_map_stats['empty_contact_map']}", "yellow")
        
        if contact_map_stats["with_contact_map"] > 0:
            # 形状统计
            shapes = contact_map_stats["contact_map_shapes"]
            unique_shapes = list(set([str(s) for s in shapes]))
            cprint(f"\nContact Map 形状:", "white")
            for shape_str in unique_shapes:
                count = sum([1 for s in shapes if str(s) == shape_str])
                cprint(f"  {shape_str}: {count} 个样本", "white")
            
            # 接触值统计
            mean_values = np.array(contact_map_stats["mean_contact_values"])
            max_values = np.array(contact_map_stats["max_contact_values"])
            
            cprint(f"\n接触值统计 (Mean):", "white")
            cprint(f"  平均: {mean_values.mean():.4f}", "white")
            cprint(f"  标准差: {mean_values.std():.4f}", "white")
            cprint(f"  最小: {mean_values.min():.4f}", "white")
            cprint(f"  最大: {mean_values.max():.4f}", "white")
            
            cprint(f"\n接触值统计 (Max):", "white")
            cprint(f"  平均: {max_values.mean():.4f}", "white")
            cprint(f"  标准差: {max_values.std():.4f}", "white")
            cprint(f"  最小: {max_values.min():.4f}", "white")
            cprint(f"  最大: {max_values.max():.4f}", "white")
        
        # 显示前 3 个样本的详细信息
        cprint(f"\n{'='*60}", "white")
        cprint(f"前 3 个样本的详细信息", "white")
        cprint(f"{'='*60}", "white")
        
        for idx in range(min(3, len(data))):
            if idx not in data:
                continue
            
            sample = data[idx]
            cprint(f"\n样本 {idx}:", "cyan")
            cprint(f"  Object: {sample.get('target_object_name', 'N/A')}", "white")
            cprint(f"  Grasp Type: {sample.get('grasp_type', 'N/A')}", "white")
            cprint(f"  Shadow qpos shape: {sample.get('shadow_qpos', np.array([])).shape}", "white")
            
            if "contact_map" in sample:
                contact_map = sample["contact_map"]
                if len(contact_map) > 0:
                    cprint(f"  Contact Map shape: {contact_map.shape}", "green")
                    cprint(f"  Contact Map mean: {contact_map.mean():.4f}", "white")
                    cprint(f"  Contact Map max: {contact_map.max():.4f}", "white")
                    cprint(f"  Contact Map min: {contact_map.min():.4f}", "white")
                    cprint(f"  Strong contacts (>0.5): {(contact_map > 0.5).sum()}", "white")
                    cprint(f"  Moderate contacts (0.3-0.5): {((contact_map > 0.3) & (contact_map <= 0.5)).sum()}", "white")
                    cprint(f"  Weak contacts (<0.3): {(contact_map <= 0.3).sum()}", "white")
                else:
                    cprint(f"  Contact Map: EMPTY", "yellow")
            else:
                cprint(f"  Contact Map: NOT FOUND", "red")
            
            if "robot_pose" in sample:
                robot_poses = sample["robot_pose"]
                if len(robot_poses) > 0:
                    cprint(f"  Robot pose count: {len(robot_poses)}", "white")
                    cprint(f"  First robot pose shape: {robot_poses[0].shape}", "white")
                else:
                    cprint(f"  Robot pose: EMPTY", "yellow")
        
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"测试完成！", "green")
        cprint(f"{'='*60}\n", "cyan")
        
        return True
        
    except Exception as e:
        cprint(f"\n✗ 加载数据失败: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


def test_contact_map_computation():
    """
    测试 contact map 计算函数（不依赖完整数据集）
    """
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"测试 Contact Map 计算函数", "cyan")
    cprint(f"{'='*60}\n", "cyan")
    
    try:
        import torch
        cprint("✓ PyTorch 可用", "green")
        
        # 创建测试数据
        hand_points = np.random.rand(100, 3)  # 100 个手部表面点
        obj_points = np.random.rand(50, 3)    # 50 个物体点
        obj_normals = np.random.rand(50, 3)   # 50 个物体法线
        obj_normals = obj_normals / np.linalg.norm(obj_normals, axis=1, keepdims=True)  # 归一化
        
        cprint(f"手部表面点: {hand_points.shape}", "white")
        cprint(f"物体点云: {obj_points.shape}", "white")
        cprint(f"物体法线: {obj_normals.shape}", "white")
        
        # 导入计算函数（从父目录）
        from store_dexonomy_retarget import compute_contact_map
        
        # 测试 PyTorch 版本
        cprint(f"\n测试 PyTorch 版本...", "white")
        import time
        start_time = time.time()
        contact_map_torch = compute_contact_map(
            hand_points, obj_points, obj_normals, use_torch=True
        )
        torch_time = time.time() - start_time
        cprint(f"✓ PyTorch 版本完成，耗时 {torch_time:.4f}s", "green")
        cprint(f"  结果形状: {contact_map_torch.shape}", "white")
        cprint(f"  平均值: {contact_map_torch.mean():.4f}", "white")
        cprint(f"  最大值: {contact_map_torch.max():.4f}", "white")
        
        # 测试 NumPy 版本
        cprint(f"\n测试 NumPy 版本...", "white")
        start_time = time.time()
        contact_map_numpy = compute_contact_map(
            hand_points, obj_points, obj_normals, use_torch=False
        )
        numpy_time = time.time() - start_time
        cprint(f"✓ NumPy 版本完成，耗时 {numpy_time:.4f}s", "green")
        cprint(f"  结果形状: {contact_map_numpy.shape}", "white")
        cprint(f"  平均值: {contact_map_numpy.mean():.4f}", "white")
        cprint(f"  最大值: {contact_map_numpy.max():.4f}", "white")
        
        # 比较结果
        diff = np.abs(contact_map_torch - contact_map_numpy).max()
        cprint(f"\n结果差异 (max abs diff): {diff:.6f}", "white")
        if diff < 1e-4:
            cprint(f"✓ PyTorch 和 NumPy 结果一致", "green")
        else:
            cprint(f"⚠ PyTorch 和 NumPy 结果有差异", "yellow")
        
        cprint(f"\n速度提升: {numpy_time / torch_time:.2f}x", "cyan")
        
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"测试完成！", "green")
        cprint(f"{'='*60}\n", "cyan")
        
        return True
        
    except ImportError as e:
        cprint(f"\n✗ 导入失败: {e}", "red")
        cprint(f"提示: 请确保已安装 torch 和相关依赖", "yellow")
        return False
    except Exception as e:
        cprint(f"\n✗ 测试失败: {e}", "red")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 contact map 功能")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="测试加载的 .npy 文件路径"
    )
    parser.add_argument(
        "--test-computation",
        action="store_true",
        help="测试 contact map 计算函数"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # 测试计算函数
    if args.test_computation:
        success = test_contact_map_computation() and success
    
    # 测试加载数据
    if args.data_path:
        success = test_load_retargeted_data(args.data_path) and success
    
    # 如果没有指定任何测试，显示帮助
    if not args.test_computation and not args.data_path:
        cprint("\n使用方法:", "cyan")
        cprint("  1. 测试计算函数: python tools/contact_map/test_contact_map.py --test-computation", "white")
        cprint("  2. 测试加载数据: python tools/contact_map/test_contact_map.py --data-path path/to/data.npy", "white")
        cprint("  3. 两者都测试: python tools/contact_map/test_contact_map.py --test-computation --data-path path/to/data.npy\n", "white")
    
    if success:
        cprint("所有测试通过！✓", "green")
    else:
        cprint("部分测试失败！✗", "red")

