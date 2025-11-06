"""
可视化 contact map 的脚本
显示接触强度的分布和统计信息
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from termcolor import cprint


def visualize_contact_map_distribution(
    contact_map: np.ndarray,
    title: str = "Contact Map Distribution",
    save_path: Optional[str] = None
):
    """
    可视化 contact map 的分布（直方图）
    
    Args:
        contact_map: (N,) contact map 数组
        title: 图表标题
        save_path: 保存路径（None 表示显示但不保存）
    """
    if len(contact_map) == 0:
        cprint("[WARNING] Contact map is empty, cannot visualize", "yellow")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    ax1 = axes[0]
    ax1.hist(contact_map, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Contact Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{title} - Histogram')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Mean: {contact_map.mean():.4f}\n"
    stats_text += f"Std: {contact_map.std():.4f}\n"
    stats_text += f"Min: {contact_map.min():.4f}\n"
    stats_text += f"Max: {contact_map.max():.4f}\n"
    stats_text += f"Median: {np.median(contact_map):.4f}"
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 累积分布
    ax2 = axes[1]
    sorted_values = np.sort(contact_map)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax2.plot(sorted_values, cumulative, linewidth=2)
    ax2.set_xlabel('Contact Value')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'{title} - CDF')
    ax2.grid(True, alpha=0.3)
    
    # 添加阈值线
    thresholds = [0.3, 0.5, 0.7]
    colors = ['green', 'orange', 'red']
    for threshold, color in zip(thresholds, colors):
        ax2.axvline(threshold, color=color, linestyle='--', alpha=0.7, label=f'Threshold {threshold}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"✓ Saved visualization to {save_path}", "green")
    else:
        plt.show()
    
    plt.close()


def visualize_contact_map_heatmap(
    contact_map: np.ndarray,
    reshape_to: Optional[tuple] = None,
    title: str = "Contact Map Heatmap",
    save_path: Optional[str] = None
):
    """
    可视化 contact map 为热力图
    
    Args:
        contact_map: (N,) contact map 数组
        reshape_to: 重塑形状 (rows, cols)，None 表示自动推断
        title: 图表标题
        save_path: 保存路径（None 表示显示但不保存）
    """
    if len(contact_map) == 0:
        cprint("[WARNING] Contact map is empty, cannot visualize", "yellow")
        return
    
    # 自动推断形状（尽量接近正方形）
    if reshape_to is None:
        n = len(contact_map)
        rows = int(np.sqrt(n))
        cols = (n + rows - 1) // rows
        # 填充到矩形
        padded = np.zeros(rows * cols)
        padded[:n] = contact_map
        contact_map_2d = padded.reshape(rows, cols)
    else:
        contact_map_2d = contact_map.reshape(reshape_to)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax.imshow(contact_map_2d, cmap='hot', aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Index (dim 2)')
    ax.set_ylabel('Index (dim 1)')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Contact Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        cprint(f"✓ Saved heatmap to {save_path}", "green")
    else:
        plt.show()
    
    plt.close()


def analyze_contact_patterns(data_dict: dict, max_samples: int = 10):
    """
    分析多个样本的 contact map 模式
    
    Args:
        data_dict: 加载的数据字典
        max_samples: 最大分析样本数
    """
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"分析 Contact Map 模式", "cyan")
    cprint(f"{'='*60}\n", "cyan")
    
    all_contact_maps = []
    sample_info = []
    
    for idx, sample in data_dict.items():
        if len(all_contact_maps) >= max_samples:
            break
        
        if "contact_map" in sample and len(sample["contact_map"]) > 0:
            all_contact_maps.append(sample["contact_map"])
            sample_info.append({
                "idx": idx,
                "object": sample.get("target_object_name", "N/A"),
                "grasp_type": sample.get("grasp_type", "N/A"),
            })
    
    if len(all_contact_maps) == 0:
        cprint("✗ No valid contact maps found", "red")
        return
    
    cprint(f"✓ Found {len(all_contact_maps)} valid contact maps", "green")
    
    # 计算统计信息
    mean_values = [cm.mean() for cm in all_contact_maps]
    max_values = [cm.max() for cm in all_contact_maps]
    strong_contacts = [(cm > 0.5).sum() for cm in all_contact_maps]
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均接触值对比
    ax1 = axes[0, 0]
    x_pos = np.arange(len(mean_values))
    ax1.bar(x_pos, mean_values, alpha=0.7, color='blue')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Mean Contact Value')
    ax1.set_title('Mean Contact Value Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{info['idx']}" for info in sample_info], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. 最大接触值对比
    ax2 = axes[0, 1]
    ax2.bar(x_pos, max_values, alpha=0.7, color='red')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Max Contact Value')
    ax2.set_title('Max Contact Value Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{info['idx']}" for info in sample_info], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. 强接触点数量对比
    ax3 = axes[1, 0]
    ax3.bar(x_pos, strong_contacts, alpha=0.7, color='green')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Number of Strong Contacts (>0.5)')
    ax3.set_title('Strong Contact Points Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{info['idx']}" for info in sample_info], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 按抓取类型分组
    ax4 = axes[1, 1]
    grasp_types = {}
    for info, mean_val in zip(sample_info, mean_values):
        grasp_type = info["grasp_type"]
        if grasp_type not in grasp_types:
            grasp_types[grasp_type] = []
        grasp_types[grasp_type].append(mean_val)
    
    grasp_type_names = list(grasp_types.keys())
    grasp_type_means = [np.mean(grasp_types[gt]) for gt in grasp_type_names]
    
    ax4.bar(range(len(grasp_type_names)), grasp_type_means, alpha=0.7, color='purple')
    ax4.set_xlabel('Grasp Type')
    ax4.set_ylabel('Mean Contact Value')
    ax4.set_title('Mean Contact by Grasp Type')
    ax4.set_xticks(range(len(grasp_type_names)))
    ax4.set_xticklabels(grasp_type_names, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # 保存到当前目录（相对于脚本位置）
    output_path = Path(__file__).parent / "contact_map_analysis.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    cprint(f"✓ Saved analysis to {output_path}", "green")
    plt.close()
    
    # 打印详细信息
    cprint(f"\n详细统计信息:", "white")
    for i, info in enumerate(sample_info):
        cprint(f"\n样本 {info['idx']}:", "cyan")
        cprint(f"  Object: {info['object']}", "white")
        cprint(f"  Grasp Type: {info['grasp_type']}", "white")
        cprint(f"  Mean Contact: {mean_values[i]:.4f}", "white")
        cprint(f"  Max Contact: {max_values[i]:.4f}", "white")
        cprint(f"  Strong Contacts: {strong_contacts[i]}", "white")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化 contact map")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="加载的 .npy 文件路径"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="要可视化的样本索引（默认：0）"
    )
    parser.add_argument(
        "--analyze-all",
        action="store_true",
        help="分析所有样本的 contact map 模式"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="分析的最大样本数（默认：10）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：./contact_map_visualizations，相对于脚本位置）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录（相对于脚本位置）
    if args.output_dir is None:
        output_dir = Path(__file__).parent / "contact_map_visualizations"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载数据
    cprint(f"\n加载数据文件: {args.data_path}", "cyan")
    try:
        data = np.load(args.data_path, allow_pickle=True).item()
        cprint(f"✓ 成功加载 {len(data)} 个样本", "green")
    except Exception as e:
        cprint(f"✗ 加载失败: {e}", "red")
        return
    
    # 分析所有样本
    if args.analyze_all:
        analyze_contact_patterns(data, max_samples=args.max_samples)
        return
    
    # 可视化单个样本
    if args.sample_idx not in data:
        cprint(f"✗ 样本索引 {args.sample_idx} 不存在", "red")
        return
    
    sample = data[args.sample_idx]
    
    if "contact_map" not in sample or len(sample["contact_map"]) == 0:
        cprint(f"✗ 样本 {args.sample_idx} 没有有效的 contact map", "red")
        return
    
    contact_map = sample["contact_map"]
    object_name = sample.get("target_object_name", "unknown")
    grasp_type = sample.get("grasp_type", "unknown")
    
    cprint(f"\n可视化样本 {args.sample_idx}:", "cyan")
    cprint(f"  Object: {object_name}", "white")
    cprint(f"  Grasp Type: {grasp_type}", "white")
    cprint(f"  Contact Map Shape: {contact_map.shape}", "white")
    cprint(f"  Mean Contact: {contact_map.mean():.4f}", "white")
    cprint(f"  Max Contact: {contact_map.max():.4f}", "white")
    
    # 生成可视化
    title_prefix = f"Sample {args.sample_idx} ({object_name}, {grasp_type})"
    
    # 分布图
    visualize_contact_map_distribution(
        contact_map,
        title=f"{title_prefix} - Distribution",
        save_path=str(output_dir / f"contact_map_{args.sample_idx}_dist.png")
    )
    
    # 热力图
    visualize_contact_map_heatmap(
        contact_map,
        title=f"{title_prefix} - Heatmap",
        save_path=str(output_dir / f"contact_map_{args.sample_idx}_heatmap.png")
    )
    
    cprint(f"\n✓ 可视化完成！输出目录: {output_dir}", "green")


if __name__ == "__main__":
    main()

