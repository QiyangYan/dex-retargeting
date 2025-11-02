"""
列出Dexonomy数据集中的可用物体和抓取类型
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dexonomy_dataset import DexonomyGraspDataset
from termcolor import cprint

def list_objects_and_grasps():
    """列出数据集中的物体和抓取信息"""
    
    # 加载数据集
    dataset = DexonomyGraspDataset(
        data_root=Path("/home/guizhewei/guizhewei/Dexonomy_dataset"),
        split="train"
    )
    
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"Dexonomy数据集物体列表", "cyan", attrs=["bold"])
    cprint(f"{'='*60}\n", "cyan")
    
    # 统计物体和抓取类型
    object_grasps = defaultdict(set)  # {object_id: set of grasp_types}
    grasp_objects = defaultdict(set)  # {grasp_type: set of object_ids}
    
    cprint("[INFO] 正在分析数据集...", "yellow")
    
    # 只分析前10000个样本以加快速度
    max_samples = min(10000, len(dataset))
    for idx in range(max_samples):
        item = dataset.grasp_index[idx]
        obj_id = item['object_id']
        grasp_type = item['grasp_type']
        
        object_grasps[obj_id].add(grasp_type)
        grasp_objects[grasp_type].add(obj_id)
    
    # 显示统计信息
    cprint(f"\n[统计信息] (基于前 {max_samples} 个样本):", "green")
    cprint(f"  不同物体数: {len(object_grasps)}", "white")
    cprint(f"  不同抓取类型数: {len(grasp_objects)}", "white")
    
    # 显示抓取类型及其物体数
    cprint(f"\n[抓取类型列表]:", "green")
    for grasp_type in sorted(grasp_objects.keys()):
        num_objects = len(grasp_objects[grasp_type])
        cprint(f"  {grasp_type:<30} {num_objects:>4} 个物体", "white")
    
    # 显示一些物体示例
    cprint(f"\n[物体示例] (前20个物体):", "green")
    for idx, (obj_id, grasps) in enumerate(list(object_grasps.items())[:20]):
        num_grasps = len(grasps)
        grasp_list = ', '.join(list(grasps)[:3])
        if num_grasps > 3:
            grasp_list += f", ... (+{num_grasps-3})"
        cprint(f"  {idx+1:2}. {obj_id[:20]}... - {num_grasps:2} 种抓取", "white")
        cprint(f"      [{grasp_list}]", "cyan")
    
    # 使用建议
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"使用示例", "cyan", attrs=["bold"])
    cprint(f"{'='*60}\n", "cyan")
    
    # 随机选择一个物体
    example_obj_id = list(object_grasps.keys())[5]
    example_grasp = list(object_grasps[example_obj_id])[0]
    
    cprint("[方法1] 指定物体ID和抓取类型:", "yellow")
    print(f"python visualize_dexonomy_grasp.py \\")
    print(f"    --object_id '{example_obj_id}' \\")
    print(f"    --grasp_type '{example_grasp}' \\")
    print(f"    --num_grasps 3")
    
    cprint("\n[方法2] 使用数据索引:", "yellow")
    print(f"python visualize_dexonomy_grasp.py --data_idx 0")
    print(f"python visualize_dexonomy_grasp.py --data_idx 50")
    print(f"python visualize_dexonomy_grasp.py --data_idx 100")
    
    cprint("\n[方法3] 按抓取类型筛选:", "yellow")
    example_types = list(grasp_objects.keys())[:3]
    for gtype in example_types:
        print(f"python visualize_dexonomy_grasp.py --grasp_type '{gtype}'")
    
    # 导出物体列表
    output_file = Path(__file__).parent / "object_list.txt"
    with open(output_file, 'w') as f:
        f.write("# Dexonomy物体列表\n\n")
        for obj_id in sorted(object_grasps.keys()):
            grasps = object_grasps[obj_id]
            f.write(f"{obj_id}\t{len(grasps)} grasps\t{', '.join(sorted(grasps))}\n")
    
    cprint(f"\n[INFO] 完整物体列表已保存到: {output_file}", "green")
    cprint(f"{'='*60}\n", "cyan")


if __name__ == "__main__":
    list_objects_and_grasps()

