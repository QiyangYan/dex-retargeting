"""
Visualize Dexonomy grasp dataset with Shadow Hand and OmniHand in SAPIEN
支持叠加显示 contact map 可视化，并自动生成 contact map 值分布统计图

新功能：
    - 自动生成 contact map 值分布直方图和累积分布函数（CDF）
    - 在控制台打印详细的统计信息（min, max, mean, std, percentiles）
    - 分析不同接触强度区域的点数分布
    - 统计图使用与可视化相同的颜色映射，便于对比

Example usage:
    # Visualize Shadow Hand only
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset
    
    # Visualize Shadow Hand and OmniHand
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --robots shadow omni
    
    # Visualize with contact map overlay (会自动生成统计图)
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy --contact_map_idx 0 --show_contact_map True
    
    # Visualize specific grasp type
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --grasp_type "1_Large_Diameter"
    
    # Visualize specific object
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --object_id "0004e882b42844eda8b4ab7379cb04c8"
    
输出文件：
    - contact_map_distribution_visualization.png: contact map 值分布统计图
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import tyro
from termcolor import cprint
import sapien
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口

# 添加父目录到路径，以便导入模块
# 当前文件: tools/contact_map/visualize_dexonomy_grasp.py
# 需要导入: position_retargeting/ 下的模块
current_dir = Path(__file__).parent  # tools/contact_map/
parent_dir = current_dir.parent      # tools/
grandparent_dir = parent_dir.parent  # position_retargeting/
sys.path.insert(0, str(grandparent_dir))

from dexonomy_dataset import DexonomyGraspDataset
from dexonomy_viewer import DexonomyGraspSAPIENViewer
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig

# 导入 contact map 相关函数
try:
    from store_dexonomy_retarget import (
        get_hand_surface_points_from_sapien,
        get_hand_surface_points_from_handmodel
    )
except ImportError as e:
    cprint(f"[WARNING] Could not import surface point extraction functions: {e}", "yellow")
    cprint("[WARNING] Contact map visualization may not work", "yellow")
    get_hand_surface_points_from_sapien = None
    get_hand_surface_points_from_handmodel = None

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

DEFAULT_SHADOWHAND_PART_NAMES = ["palm", "thumb", "index", "middle", "ring", "pinky"]


def load_contact_map_data(contact_map_path: str, contact_map_idx: int) -> Optional[Dict]:
    """
    加载 contact map 数据
    
    Args:
        contact_map_path: contact map 数据文件路径 (.npy)
        contact_map_idx: 数据索引
    
    Returns:
        contact_map 数据字典，如果加载失败返回 None
    """
    try:
        data = np.load(contact_map_path, allow_pickle=True).item()
        if contact_map_idx not in data:
            cprint(f"[WARNING] Contact map index {contact_map_idx} not found in data", "yellow")
            cprint(f"[INFO] Available indices: {sorted(list(data.keys()))[:10]}... (showing first 10)", "white")
            return None
        
        sample = data[contact_map_idx]
        
        # 检查物体点云的 contact map
        if "contact_map_object" not in sample:
            cprint(f"[WARNING] Key 'contact_map_object' not found in sample", "yellow")
            cprint(f"[INFO] Available keys: {list(sample.keys())}", "white")
            return None
        
        contact_map_object = sample["contact_map_object"]
        if len(contact_map_object) == 0:
            cprint(f"[WARNING] Contact map (object) is empty for index {contact_map_idx}", "yellow")
            return None
        
        # 检查 object_point_cloud
        if "object_point_cloud" not in sample:
            cprint(f"[WARNING] Key 'object_point_cloud' not found in sample", "yellow")
            return None
        
        object_point_cloud = sample["object_point_cloud"]
        if len(object_point_cloud) == 0:
            cprint(f"[WARNING] Object point cloud is empty for index {contact_map_idx}", "yellow")
            return None
        
        # 检查数据形状和类型
        cprint(f"[INFO] Loaded contact map data:", "green")
        cprint(f"  contact_map_object: shape={contact_map_object.shape}, dtype={contact_map_object.dtype}, "
               f"min={contact_map_object.min():.6f}, max={contact_map_object.max():.6f}, mean={contact_map_object.mean():.6f}", "white")
        cprint(f"  object_point_cloud: shape={object_point_cloud.shape}, dtype={object_point_cloud.dtype}", "white")
        
        # 确保 contact_map_object 是一维数组
        if len(contact_map_object.shape) > 1:
            cprint(f"[WARNING] contact_map_object has shape {contact_map_object.shape}, flattening...", "yellow")
            contact_map_object = contact_map_object.flatten()
            sample["contact_map_object"] = contact_map_object
        
        # 确保 object_point_cloud 是二维数组 (N, 3)
        if len(object_point_cloud.shape) == 1:
            cprint(f"[WARNING] object_point_cloud has shape {object_point_cloud.shape}, reshaping...", "yellow")
            # 尝试重塑为 (N, 3)
            if len(object_point_cloud) % 3 == 0:
                object_point_cloud = object_point_cloud.reshape(-1, 3)
                sample["object_point_cloud"] = object_point_cloud
            else:
                cprint(f"[ERROR] Cannot reshape object_point_cloud (length {len(object_point_cloud)} not divisible by 3)", "red")
                return None
        
        # 检查长度是否匹配
        if len(contact_map_object) != len(object_point_cloud):
            cprint(f"[WARNING] Length mismatch: contact_map_object={len(contact_map_object)}, object_point_cloud={len(object_point_cloud)}", "yellow")
            # 使用较小的长度
            min_len = min(len(contact_map_object), len(object_point_cloud))
            sample["contact_map_object"] = contact_map_object[:min_len]
            sample["object_point_cloud"] = object_point_cloud[:min_len]
            cprint(f"[INFO] Trimmed both arrays to length {min_len}", "white")

        # 处理分部位的 contact map（如果存在）
        if "contact_map_object_parts" in sample:
            part_maps_raw = sample["contact_map_object_parts"]
            try:
                if isinstance(part_maps_raw, list):
                    part_maps = [np.asarray(pm, dtype=np.float32) for pm in part_maps_raw]
                    contact_map_object_parts = np.stack(part_maps, axis=0)
                else:
                    contact_map_object_parts = np.asarray(part_maps_raw)
                    if contact_map_object_parts.dtype != np.float32:
                        contact_map_object_parts = contact_map_object_parts.astype(np.float32)
            except Exception as e:
                cprint(f"[WARNING] Failed to parse contact_map_object_parts ({type(part_maps_raw)}): {e}", "yellow")
                contact_map_object_parts = None

            if contact_map_object_parts is not None:
                if contact_map_object_parts.ndim != 2:
                    cprint(f"[WARNING] contact_map_object_parts has unexpected shape {contact_map_object_parts.shape}, reshaping...", "yellow")
                    contact_map_object_parts = contact_map_object_parts.reshape(contact_map_object_parts.shape[0], -1)

                part_names_raw = sample.get("contact_map_object_part_names", DEFAULT_SHADOWHAND_PART_NAMES)
                if isinstance(part_names_raw, np.ndarray):
                    part_names = [str(name) for name in part_names_raw.tolist()]
                elif isinstance(part_names_raw, list):
                    part_names = [str(name) for name in part_names_raw]
                else:
                    part_names = DEFAULT_SHADOWHAND_PART_NAMES.copy()

                if contact_map_object_parts.shape[0] != len(part_names):
                    cprint(f"[WARNING] Part count mismatch: maps={contact_map_object_parts.shape[0]}, names={len(part_names)}", "yellow")
                    min_parts = min(contact_map_object_parts.shape[0], len(part_names))
                    contact_map_object_parts = contact_map_object_parts[:min_parts]
                    part_names = part_names[:min_parts]

                if contact_map_object_parts.shape[1] != sample["object_point_cloud"].shape[0]:
                    cprint(f"[WARNING] Part contact map length mismatch: {contact_map_object_parts.shape[1]} vs object cloud {sample['object_point_cloud'].shape[0]}", "yellow")
                    min_len = min(contact_map_object_parts.shape[1], sample["object_point_cloud"].shape[0])
                    contact_map_object_parts = contact_map_object_parts[:, :min_len]
                    sample["contact_map_object"] = sample["contact_map_object"][:min_len]
                    sample["object_point_cloud"] = sample["object_point_cloud"][:min_len]
                    cprint(f"[INFO] Trimmed part contact maps and arrays to length {min_len}", "white")

                sample["contact_map_object_parts"] = contact_map_object_parts
                sample["contact_map_object_part_names"] = part_names
                cprint(f"[INFO] Loaded contact map parts: {part_names}", "green")
            else:
                sample.pop("contact_map_object_parts", None)
                sample.pop("contact_map_object_part_names", None)
        
        return sample
    except Exception as e:
        cprint(f"[ERROR] Failed to load contact map: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


def plot_contact_map_distribution(
    contact_map: np.ndarray,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    fig_size: tuple = (12, 5),
):
    """
    绘制 contact map 值的分布统计图
    
    Args:
        contact_map: (N,) contact map 值数组
        save_path: 保存图片的路径（如果为 None，则保存到当前目录）
        show_plot: 是否在控制台显示统计信息
        fig_size: 图片大小（宽, 高）英寸
    """
    if len(contact_map) == 0:
        cprint("[WARNING] Contact map is empty, cannot plot distribution", "yellow")
        return
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    # 子图1：直方图
    ax1 = axes[0]
    n, bins, patches = ax1.hist(contact_map, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    # 根据值的范围给直方图条着色（使用自适应颜色映射）
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # 使用实际数据范围进行归一化，与 3D 可视化保持一致
    data_min = contact_map.min()
    data_max = contact_map.max()
    col_norm = (bin_centers - data_min) / (data_max - data_min + 1e-8)
    
    for c, p, norm_val in zip(bin_centers, patches, col_norm):
        # 使用与球体相同的颜色映射：蓝色 -> 青色 -> 绿色 -> 黄色 -> 红色
        # 这样直方图的颜色与 3D 可视化完全一致
        norm_val = np.clip(norm_val, 0.0, 1.0)
        if norm_val < 0.25:
            r, g, b = 0.0, norm_val * 4.0, 1.0
        elif norm_val < 0.5:
            r, g, b = 0.0, 1.0, 1.0 - (norm_val - 0.25) * 4.0
        elif norm_val < 0.75:
            r, g, b = (norm_val - 0.5) * 4.0, 1.0, 0.0
        else:
            r, g, b = 1.0, 1.0 - (norm_val - 0.75) * 4.0, 0.0
        p.set_facecolor((r, g, b))
    
    ax1.set_xlabel('Contact Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    title_text = f'Contact Map Value Distribution\n(Color: {data_min:.4f} 🔵 Blue → {data_max:.4f} 🔴 Red)'
    ax1.set_title(title_text, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息文本
    stats_text = f'Total Points: {len(contact_map)}\n'
    stats_text += f'Min: {contact_map.min():.4f}\n'
    stats_text += f'Max: {contact_map.max():.4f}\n'
    stats_text += f'Mean: {contact_map.mean():.4f}\n'
    stats_text += f'Std: {contact_map.std():.4f}\n'
    stats_text += f'Median: {np.median(contact_map):.4f}'
    
    ax1.text(0.98, 0.97, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图2：累积分布函数（CDF）
    ax2 = axes[1]
    sorted_values = np.sort(contact_map)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax2.plot(sorted_values, cumulative, linewidth=2, color='darkblue')
    ax2.fill_between(sorted_values, cumulative, alpha=0.3, color='steelblue')
    
    ax2.set_xlabel('Contact Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加百分位数线
    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(contact_map, percentiles)
    for p, val in zip(percentiles, percentile_values):
        ax2.axvline(val, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(val, 0.02, f'P{p}\n{val:.3f}', 
                rotation=0, fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 整体布局调整
    plt.tight_layout()
    
    # 保存图片
    if save_path is None:
        save_path = f"contact_map_distribution_{np.random.randint(10000)}.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    cprint(f"[INFO] Contact map distribution plot saved to: {save_path}", "green")
    
    # 在控制台打印统计信息
    if show_plot:
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"Contact Map Value Distribution Statistics", "cyan", attrs=["bold"])
        cprint(f"{'='*60}", "cyan")
        cprint(f"Total points: {len(contact_map)}", "white")
        cprint(f"Min value:    {contact_map.min():.6f}", "white")
        cprint(f"Max value:    {contact_map.max():.6f}", "white")
        cprint(f"Mean value:   {contact_map.mean():.6f}", "white")
        cprint(f"Std dev:      {contact_map.std():.6f}", "white")
        cprint(f"Median:       {np.median(contact_map):.6f}", "white")
        cprint(f"\nPercentiles:", "yellow")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(contact_map, p)
            cprint(f"  P{p:2d}: {val:.6f}", "white")
        
        # 分析接触区域
        cprint(f"\nContact Region Analysis:", "yellow")
        cprint(f"  Very low contact  (<0.1): {(contact_map < 0.1).sum():5d} points ({(contact_map < 0.1).sum()/len(contact_map)*100:5.2f}%)", "white")
        cprint(f"  Low contact   (0.1-0.3): {((contact_map >= 0.1) & (contact_map < 0.3)).sum():5d} points ({((contact_map >= 0.1) & (contact_map < 0.3)).sum()/len(contact_map)*100:5.2f}%)", "white")
        cprint(f"  Medium contact (0.3-0.5): {((contact_map >= 0.3) & (contact_map < 0.5)).sum():5d} points ({((contact_map >= 0.3) & (contact_map < 0.5)).sum()/len(contact_map)*100:5.2f}%)", "white")
        cprint(f"  High contact   (0.5-0.7): {((contact_map >= 0.5) & (contact_map < 0.7)).sum():5d} points ({((contact_map >= 0.5) & (contact_map < 0.7)).sum()/len(contact_map)*100:5.2f}%)", "white")
        cprint(f"  Very high contact (>=0.7): {(contact_map >= 0.7).sum():5d} points ({(contact_map >= 0.7).sum()/len(contact_map)*100:5.2f}%)", "white")
        cprint(f"{'='*60}\n", "cyan")
    
    plt.close()


def visualize_object_surface_points(
    viewer: DexonomyGraspSAPIENViewer,
    object_points: np.ndarray,
    object_normals: Optional[np.ndarray] = None,
    sphere_radius: float = 0.003,
    max_points: int = 5000,
    color: Optional[np.ndarray] = None,
):
    """
    可视化物体表面点
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        object_points: (M, 3) 物体表面点世界坐标
        object_normals: (M, 3) 物体表面法向量（可选，用于调试）
        sphere_radius: 球体半径（米）
        max_points: 最大显示点数（如果点数太多，进行采样）
        color: (4,) RGBA 颜色数组，默认为浅蓝色
    """
    if len(object_points) == 0:
        cprint("[WARNING] No object points to visualize", "yellow")
        return
    
    # 如果点数太多，进行采样
    if len(object_points) > max_points:
        indices = np.random.choice(len(object_points), max_points, replace=False)
        object_points = object_points[indices]
        if object_normals is not None:
            object_normals = object_normals[indices]
        cprint(f"[INFO] Sampled {max_points} points from {len(object_points)} total object points", "white")
    
    # 清除现有的 object point markers
    if not hasattr(viewer, 'object_point_markers'):
        viewer.object_point_markers = []
    else:
        for marker in viewer.object_point_markers:
            viewer.scene.remove_actor(marker)
        viewer.object_point_markers.clear()
    
    # 默认颜色：浅蓝色（与手部颜色区分）
    if color is None:
        color = np.array([0.3, 0.6, 1.0, 0.8])  # 浅蓝色 RGBA
    
    # 创建球体 markers
    cprint(f"[INFO] Creating {len(object_points)} object surface point markers...", "cyan")
    
    for i, pos in enumerate(object_points):
        # 创建球体 material
        sphere_material = sapien.render.RenderMaterial()
        sphere_material.set_base_color(color)
        sphere_material.set_roughness(0.3)
        sphere_material.set_metallic(0.1)
        sphere_material.set_specular(0.8)
        
        # 创建球体
        builder = viewer.scene.create_actor_builder()
        builder.add_sphere_visual(radius=sphere_radius, material=sphere_material)
        marker = builder.build_static(name=f"object_point_marker_{i}")
        marker.set_pose(sapien.Pose(pos))
        viewer.object_point_markers.append(marker)
    
    cprint(f"[INFO] Created {len(viewer.object_point_markers)} object surface point markers", "green")


def visualize_object_surface_points_with_contact_map(
    viewer: DexonomyGraspSAPIENViewer,
    object_points: np.ndarray,
    contact_map: np.ndarray,
    sphere_radius: float = 0.008,
    max_points: int = 5000,
    sphere_alpha: float = 0.7,
    distribution_name: Optional[str] = None,
):
    """
    在物体表面点上可视化 contact map（使用颜色编码的球体）
    
    注意：此函数只在采样的表面点位置创建彩色球体标记，不会修改物体 mesh。
    物体的原始 mesh 会保持可见。如果球体看起来覆盖了整个物体，可以：
    - 减少 max_points（使球体更稀疏）
    - 减小 sphere_radius（使球体更小）
    - 增加 sphere_alpha（使球体更透明）
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        object_points: (M, 3) 物体表面点世界坐标（采样点）
        contact_map: (M,) contact map 值数组，对应每个采样点的接触值
        sphere_radius: 球体半径（米），建议 0.003-0.010
        max_points: 最大显示点数（如果点数太多，进行采样），建议 500-5000
        sphere_alpha: 球体透明度 (0-1)，较小的值使物体 mesh 更容易看到
    """
    if len(contact_map) != len(object_points):
        cprint(f"[WARNING] Contact map size ({len(contact_map)}) != object points size ({len(object_points)})", "yellow")
        # 使用较小的长度
        min_len = min(len(contact_map), len(object_points))
        contact_map = contact_map[:min_len]
        object_points = object_points[:min_len]

    contact_values_full = np.asarray(contact_map, dtype=np.float32).copy()
    contact_values = contact_values_full.copy()
    
    # 检查 contact_map 的值范围
    cm_min_orig = float(contact_values_full.min()) if len(contact_values_full) > 0 else 0.0
    cm_max_orig = float(contact_values_full.max()) if len(contact_values_full) > 0 else 0.0
    cm_mean = float(contact_values_full.mean()) if len(contact_values_full) > 0 else 0.0
    cm_std = float(contact_values_full.std()) if len(contact_values_full) > 0 else 0.0
    cprint(f"[DEBUG] Original contact map stats: min={cm_min_orig:.6f}, max={cm_max_orig:.6f}, mean={cm_mean:.6f}, std={cm_std:.6f}", "cyan")
    
    # 🎨 使用自适应颜色映射：根据实际值范围来分配颜色
    # 这样即使值集中在小区间（如 0.08-0.22），也能看到清晰的颜色渐变
    cprint(f"[INFO] Using adaptive color mapping based on actual value range [{cm_min_orig:.6f}, {cm_max_orig:.6f}]", "green")
    
    # 将值归一化到 [0, 1] 区间，用于颜色映射
    denom = (cm_max_orig - cm_min_orig + 1e-8)
    contact_map_normalized_full = (contact_values_full - cm_min_orig) / denom
    contact_map_normalized = contact_map_normalized_full.copy()
    
    cprint(f"[DEBUG] After adaptive normalization: min={contact_map_normalized.min():.6f}, max={contact_map_normalized.max():.6f}", "cyan")
    cprint(f"[INFO] Color mapping: {cm_min_orig:.6f} (blue) -> {cm_max_orig:.6f} (red)", "yellow")
    
    # 如果点数太多，进行采样
    if len(object_points) > max_points:
        indices = np.random.choice(len(object_points), max_points, replace=False)
        object_points = object_points[indices]
        contact_values = contact_values[indices]
        contact_map_normalized = contact_map_normalized[indices]
        cprint(f"[INFO] Sampled {max_points} points from total points", "white")
    
    # 清除现有的 contact map markers
    if not hasattr(viewer, 'object_contact_map_markers'):
        viewer.object_contact_map_markers = []
    else:
        for marker in viewer.object_contact_map_markers:
            viewer.scene.remove_actor(marker)
        viewer.object_contact_map_markers.clear()
    
    # 清除 object_point_markers（如果存在），避免冲突
    if hasattr(viewer, 'object_point_markers') and len(viewer.object_point_markers) > 0:
        cprint(f"[INFO] Clearing {len(viewer.object_point_markers)} existing object point markers", "yellow")
        for marker in viewer.object_point_markers:
            viewer.scene.remove_actor(marker)
        viewer.object_point_markers.clear()
    
    # 创建颜色映射：从蓝色（低接触）到红色（高接触）
    # 使用归一化后的值（已映射到 [0, 1]）来分配颜色
    colors = np.zeros((len(contact_map_normalized), 4))  # RGBA
    
    # 使用 colormap: 蓝色(0) -> 青色(0.25) -> 绿色(0.5) -> 黄色(0.75) -> 红色(1)
    # 这样即使原始值在小区间（如 0.08-0.22），也能展现完整的颜色渐变
    for i, norm_value in enumerate(contact_map_normalized):
        # 确保归一化值在 [0, 1] 范围内
        norm_value = np.clip(norm_value, 0.0, 1.0)
        
        if norm_value < 0.25:
            # 蓝色到青色
            r = 0.0
            g = norm_value * 4.0
            b = 1.0
        elif norm_value < 0.5:
            # 青色到绿色
            r = 0.0
            g = 1.0
            b = 1.0 - (norm_value - 0.25) * 4.0
        elif norm_value < 0.75:
            # 绿色到黄色
            r = (norm_value - 0.5) * 4.0
            g = 1.0
            b = 0.0
        else:
            # 黄色到红色
            r = 1.0
            g = 1.0 - (norm_value - 0.75) * 4.0
            b = 0.0
        
        colors[i] = [r, g, b, sphere_alpha]  # RGBA，使用可调节的 alpha 值
    
    # 创建球体 markers（只在采样点上，不修改物体 mesh）
    cprint(f"[INFO] Creating {len(object_points)} contact map sphere markers on sampled surface points...", "cyan")
    cprint(f"      Sphere radius: {sphere_radius}m, Object mesh will remain visible underneath", "white")
    
    for i, (pos, color) in enumerate(zip(object_points, colors)):
        # 创建球体 material - 使用更少反光的材质以显示真实颜色
        sphere_material = sapien.render.RenderMaterial()
        sphere_material.set_base_color(color)
        sphere_material.set_roughness(0.9)  # 提高粗糙度，减少反光
        sphere_material.set_metallic(0.0)   # 移除金属效果
        sphere_material.set_specular(0.1)   # 大幅降低镜面反射
        
        # 创建球体（独立的 actor，不修改物体 mesh）
        builder = viewer.scene.create_actor_builder()
        builder.add_sphere_visual(radius=sphere_radius, material=sphere_material)
        marker = builder.build_static(name=f"object_contact_map_marker_{i}")
        marker.set_pose(sapien.Pose(pos))
        viewer.object_contact_map_markers.append(marker)
    
    cprint(f"[INFO] Created {len(viewer.object_contact_map_markers)} object contact map markers", "green")
    cprint(f"  Original contact map range: [{cm_min_orig:.6f}, {cm_max_orig:.6f}]", "white")
    cprint(f"  Original contact map mean: {cm_mean:.6f}", "white")
    cprint(f"  Color mapping: MIN={cm_min_orig:.6f} (🔵 Blue) -> MAX={cm_max_orig:.6f} (🔴 Red)", "yellow", attrs=["bold"])
    
    # 绘制 contact map 值的分布统计图（使用原始值）
    try:
        # 为了统计，需要使用完整的未采样的 contact map
        # 重新计算以获取完整数据
        cprint(f"[INFO] Generating contact map distribution plot...", "cyan")
        distribution_filename = distribution_name if distribution_name is not None else "contact_map_distribution_visualization.png"
        plot_contact_map_distribution(
            contact_values_full,
            save_path=distribution_filename,
            show_plot=True,
        )
    except Exception as e:
        cprint(f"[WARNING] Failed to plot contact map distribution: {e}", "yellow")
        import traceback
        traceback.print_exc()


def visualize_dexonomy_grasp(
    dexonomy_dir: str = "/home/guizhewei/guizhewei/Dexonomy_dataset",
    robots: Optional[List[RobotName]] = None,
    grasp_type: Optional[str] = None,
    object_id: Optional[str] = None,
    scale_name: str = "scale005",
    data_idx: int = 0,
    fps: int = 10,
    headless: bool = False,
    split: str = "train",
    num_grasps: int = 1,
    show_ground: bool = False,
    show_table: bool = False,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.vector,
    two_optimizers: bool = False,
    second_optimizer_type: str = "FINGERTIP",
    y_offset: float = 0.5,
    contact_map_path: Optional[str] = None,
    contact_map_idx: Optional[int] = None,
    show_contact_map: bool = False,
    show_object_points: bool = False,
    contact_sphere_radius: float = 0.008,
    contact_max_points: int = 5000,
    contact_sphere_alpha: float = 0.7,
    show_contact_map_parts: bool = False,
):
    """
    Visualize Dexonomy grasp dataset with Shadow Hand and optional retargeting to other hands
    支持叠加显示物体点云上的 contact map 和物体表面点可视化
    
    Args:
        dexonomy_dir: Root directory of Dexonomy dataset
        robots: List of robot hands to visualize. If None, show Shadow Hand only.
                Available: shadow, allegro, svh, omni, etc.
        grasp_type: Specific grasp type to visualize (e.g., "1_Large_Diameter").
                    If None, load all grasp types.
        object_id: Specific object ID to visualize. If None, use data_idx to select.
        scale_name: Object scale to use (e.g., "scale005", "scale008")
        data_idx: Index of grasp sample to visualize if object_id is not specified
        fps: Rendering FPS
        headless: Run in headless mode and save video
        split: Dataset split to use ("train", "test", or "all")
        num_grasps: Number of consecutive grasps to visualize starting from data_idx
        show_ground: Show ground plane
        show_table: Show table
        hand_type: HandType (right or left)
        retargeting_type: RetargetingType for first optimizer (position, vector, fingertip, dexpilot)
        two_optimizers: Whether to use two optimizers (default: False)
        second_optimizer_type: Type of second optimizer ("VECTOR" or "FINGERTIP")
        y_offset: Y-axis spacing between robots for parallel display (default: 0.5m)
        contact_map_path: Path to contact map data file (.npy)
        contact_map_idx: Index in contact map data file
        show_contact_map: Whether to show contact map overlay on object points (requires contact_map_path)
                          This visualizes contact_map_object on object_point_cloud with color coding
        show_object_points: Whether to show object surface points without contact map coloring (requires contact_map_path)
        contact_sphere_radius: Radius of contact map spheres in meters (default: 0.008m=8mm)
                               Smaller values (0.003-0.005) for less coverage, larger (0.010-0.015) for better visibility
        contact_max_points: Maximum number of contact sphere markers to display (default: 5000)
                            Reduce this (e.g., 500-1000) to make visualization sparser and see object mesh better
        contact_sphere_alpha: Transparency of contact spheres, 0-1 (default: 0.7)
        show_contact_map_parts: Sequentially visualize per-part contact maps (palm + 5 fingers) if data is available
                              Lower values (0.3-0.5) make object mesh more visible through spheres
        show_contact_map_parts: If True and per-part contact maps are available, sequentially visualize palm & each finger in SAPIEN
    
    注意：Contact map 可视化说明
        - 彩色球体只在采样的表面点上创建，不会修改物体 mesh
        - 物体的原始 mesh 会正常显示在球体下方
        - 如果球体看起来覆盖了整个物体，可以调整：
          1. 减少 contact_max_points（例如 500-1000）使球体更稀疏
          2. 减小 contact_sphere_radius（例如 0.003-0.005）使球体更小
          3. 降低 contact_sphere_alpha（例如 0.3-0.5）增加透明度
    """
    # Setup paths
    data_root = Path(dexonomy_dir).absolute()
    if not data_root.exists():
        raise ValueError(f"Dexonomy directory does not exist: {data_root}")
    
    # 当前文件: tools/contact_map/visualize_dexonomy_grasp.py
    # 目标路径: dex-retargeting/assets/robots/hands
    # 从 position_retargeting/ 向上 2 级到 dex-retargeting/
    robot_dir = Path(__file__).absolute().parent.parent.parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    # Load contact map data if provided
    contact_map_data = None
    if contact_map_path is not None and contact_map_idx is not None:
        cprint(f"\n[INFO] Loading contact map from {contact_map_path} (index {contact_map_idx})", "cyan")
        contact_map_data = load_contact_map_data(contact_map_path, contact_map_idx)
        if contact_map_data is not None:
            show_contact_map = True
            cprint(f"[INFO] Contact map will be visualized on object points", "green")
        else:
            cprint(f"[WARNING] Failed to load contact map, continuing without overlay", "yellow")
            show_contact_map = False
    
    # Default to Shadow Hand only if no robots specified
    if robots is None:
        robots = [RobotName.shadow_no_wrist]
    
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"Dexonomy Grasp Visualization", "cyan", attrs=["bold"])
    cprint(f"{'='*60}", "cyan")
    cprint(f"Dataset: {data_root}", "green")
    cprint(f"Robots: {robots}", "green")
    cprint(f"Split: {split}", "green")
    if grasp_type:
        cprint(f"Grasp Type: {grasp_type}", "green")
    if object_id:
        cprint(f"Object ID: {object_id}", "green")
    if show_contact_map:
        cprint(f"Contact Map (on Object): ENABLED", "green", attrs=["bold"])
    if show_object_points:
        cprint(f"Object Surface Points: ENABLED", "green", attrs=["bold"])
    if show_contact_map_parts:
        cprint(f"Contact Map Parts Visualization: ENABLED", "green", attrs=["bold"])
    cprint(f"{'='*60}\n", "cyan")
    
    # Load dataset
    dataset = DexonomyGraspDataset(
        data_root=data_root,
        grasp_type=grasp_type,
        object_ids=[object_id] if object_id else None,
        split=split,
    )
    
    if len(dataset) == 0:
        cprint("[ERROR] No grasp samples found in dataset!", "red")
        return
    
    # Get grasp sample
    if object_id and grasp_type:
        # Load specific grasp
        grasp_data = dataset.get_grasp_by_object_and_type(
            object_id=object_id,
            grasp_type=grasp_type,
            scale_name=scale_name,
        )
        if grasp_data is None:
            cprint(f"[ERROR] Grasp not found for object {object_id}, type {grasp_type}", "red")
            return
    else:
        # Load by index
        if data_idx >= len(dataset):
            cprint(f"[WARNING] data_idx {data_idx} >= dataset size {len(dataset)}, using 0", "yellow")
            data_idx = 0
        grasp_data = dataset[data_idx]
    
    # Print grasp information
    cprint(f"\n[INFO] Loaded grasp sample:", "cyan")
    cprint(f"  Grasp Type: {grasp_data['grasp_type']}", "yellow")
    cprint(f"  Object ID: {grasp_data['object_id']}", "yellow")
    cprint(f"  Scale: {grasp_data['scale_name']}", "yellow")
    cprint(f"  Grasp Index: {grasp_data['grasp_idx']}/{grasp_data['num_grasps_in_file']}", "yellow")
    cprint(f"  Object Mass: {grasp_data['object_info']['mass']:.3f} kg", "yellow")
    cprint(f"  Object OBB: {grasp_data['object_info']['obb']}", "yellow")
    
    # Viewer 工厂函数，便于在多次可视化（如逐指显示）时重建 viewer
    def create_viewer_instance() -> DexonomyGraspSAPIENViewer:
        return DexonomyGraspSAPIENViewer(
            robot_names=list(robots),
            headless=headless,
            use_ray_tracing=False,
            show_ground=show_ground,
            show_table=show_table,
            data_root=data_root,
            hand_type=hand_type,
            retargeting_type=retargeting_type,
            two_optimizers=two_optimizers,
            second_optimizer_type=second_optimizer_type,
        )

    viewer: Optional[DexonomyGraspSAPIENViewer] = None
    
    contact_map_parts = None
    contact_map_part_names: Optional[List[str]] = None
    if contact_map_data is not None:
        if "contact_map_object_parts" in contact_map_data:
            contact_map_parts = np.asarray(contact_map_data["contact_map_object_parts"])
            if contact_map_parts.ndim == 1:
                contact_map_parts = contact_map_parts.reshape(1, -1)
            contact_map_part_names_raw = contact_map_data.get("contact_map_object_part_names", DEFAULT_SHADOWHAND_PART_NAMES)
            if isinstance(contact_map_part_names_raw, np.ndarray):
                contact_map_part_names = [str(name) for name in contact_map_part_names_raw.tolist()]
            elif isinstance(contact_map_part_names_raw, list):
                contact_map_part_names = [str(name) for name in contact_map_part_names_raw]
            else:
                contact_map_part_names = DEFAULT_SHADOWHAND_PART_NAMES.copy()
            if contact_map_parts.shape[0] != len(contact_map_part_names):
                min_parts = min(contact_map_parts.shape[0], len(contact_map_part_names))
                contact_map_parts = contact_map_parts[:min_parts]
                contact_map_part_names = contact_map_part_names[:min_parts]
            cprint(f"[INFO] Contact map per-part data detected: {contact_map_part_names}", "green")

    # Helper function to render with contact map and/or object points overlay
    def render_with_contact_map(
        grasp_data_item,
        contact_map_override: Optional[np.ndarray] = None,
        part_name: Optional[str] = None,
        part_index: Optional[int] = None,
        total_parts: Optional[int] = None,
    ):
        """渲染单个抓取，可选地叠加整体或分部位的 contact map"""
        nonlocal viewer
        if viewer is None or (hasattr(viewer, "viewer") and viewer.viewer.closed):
            viewer = create_viewer_instance()

        if part_name is not None:
            if total_parts is not None and part_index is not None:
                cprint(f"\n[INFO] Visualizing contact map for part {part_index+1}/{total_parts}: {part_name}", "cyan", attrs=["bold"])
            else:
                cprint(f"\n[INFO] Visualizing contact map for part: {part_name}", "cyan", attrs=["bold"])

        # 如果启用了 contact map 或 object points，先设置手部姿态
        if (show_contact_map or show_object_points or contact_map_override is not None) and contact_map_data is not None:
            try:
                # 加载物体
                viewer.load_object(grasp_data_item)
                
                # 设置 Shadow Hand 姿态
                grasp_qpos = grasp_data_item["grasp_qpos"]
                viewer.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
                
                # 可视化物体点云上的 contact map
                if contact_map_override is not None or show_contact_map:
                    if "object_point_cloud" in contact_map_data and (
                        contact_map_override is not None or "contact_map_object" in contact_map_data
                    ):
                        object_points = contact_map_data["object_point_cloud"]
                        contact_map_object = (
                            contact_map_override
                            if contact_map_override is not None
                            else contact_map_data["contact_map_object"]
                        )
                        
                        if len(object_points) > 0 and len(contact_map_object) > 0:
                            # 使用 contact map 值来着色物体点云
                            # 只在采样点上创建彩色球体，物体 mesh 保持原样
                            if part_name is not None:
                                safe_part = part_name.replace(" ", "_")
                                distribution_file = f"contact_map_distribution_{safe_part}.png"
                            else:
                                distribution_file = "contact_map_distribution_visualization.png"

                            visualize_object_surface_points_with_contact_map(
                                viewer,
                                object_points,
                                contact_map_object,
                                sphere_radius=contact_sphere_radius,
                                max_points=contact_max_points,
                                sphere_alpha=contact_sphere_alpha,
                                distribution_name=distribution_file,
                            )
                            if part_name is not None:
                                cprint(f"[INFO] Contact map visualization for '{part_name}' added to scene", "green")
                            else:
                                cprint(f"[INFO] Contact map visualization on object added to scene", "green")
                        else:
                            cprint(f"[WARNING] No object point cloud or contact map data available", "yellow")
                    else:
                        cprint(f"[WARNING] Missing object_point_cloud or contact_map_object in data", "yellow")
                
                # 可视化物体表面点（不带 contact map 颜色）
                if show_object_points:
                    if "object_point_cloud" in contact_map_data and len(contact_map_data["object_point_cloud"]) > 0:
                        object_points = contact_map_data["object_point_cloud"]
                        object_normals = contact_map_data.get("object_normal_cloud", None)
                        visualize_object_surface_points(
                            viewer,
                            object_points,
                            object_normals=object_normals,
                            sphere_radius=0.003,
                            max_points=5000,
                            color=np.array([0.3, 0.6, 1.0, 0.8])  # 浅蓝色
                        )
                        cprint(f"[INFO] Object surface points visualization added to scene", "green")
                    else:
                        cprint(f"[WARNING] No object point cloud in contact map data", "yellow")
                        
            except Exception as e:
                cprint(f"[ERROR] Failed to visualize contact map/object points: {e}", "red")
                import traceback
                traceback.print_exc()
        
        # 调用原始的渲染函数
        viewer.render_grasp_single(grasp_data_item, fps=fps, y_offset=y_offset)
    
    # Render grasp(s)
    if num_grasps == 1:
        if show_contact_map_parts:
            if contact_map_parts is None or contact_map_parts.size == 0:
                cprint("[WARNING] show_contact_map_parts=True but per-part contact map data is unavailable. Showing overall contact map instead.", "yellow")
            else:
                cprint("\n[INFO] Sequentially visualizing contact map for palm & each finger. Close the viewer window to proceed to the next part.", "cyan")
                total_parts = contact_map_parts.shape[0]
                for idx in range(total_parts):
                    part_map = contact_map_parts[idx]
                    part_name = contact_map_part_names[idx] if contact_map_part_names and idx < len(contact_map_part_names) else f"part_{idx}"
                    if part_map.size == 0 or np.allclose(part_map, 0.0):
                        cprint(f"[INFO] Skipping part '{part_name}' (no contact points).", "yellow")
                        continue
                    render_with_contact_map(
                        grasp_data,
                        contact_map_override=part_map,
                        part_name=part_name,
                        part_index=idx,
                        total_parts=total_parts,
                    )
                    viewer = None  # 下一次循环重建 viewer
                return

        # Render single grasp (overall contact map or none)
        render_with_contact_map(grasp_data)
    else:
        # Render multiple consecutive grasps
        cprint(f"\n[INFO] Rendering {num_grasps} consecutive grasps starting from index {data_idx}", "cyan")
        if show_contact_map_parts:
            cprint("[WARNING] show_contact_map_parts currently supports num_grasps == 1. Falling back to overall contact map visualization.", "yellow")
        for i in range(num_grasps):
            current_idx = data_idx + i
            if current_idx >= len(dataset):
                cprint(f"[WARNING] Reached end of dataset at index {current_idx}", "yellow")
                break
            
            grasp_data = dataset[current_idx]
            cprint(f"\n{'='*60}", "cyan")
            cprint(f"Grasp {i+1}/{num_grasps} (Dataset Index: {current_idx})", "cyan", attrs=["bold"])
            cprint(f"{'='*60}", "cyan")
            cprint(f"  Object: {grasp_data['object_id'][:20]}...", "yellow")
            cprint(f"  Type: {grasp_data['grasp_type']}", "yellow")
            cprint(f"  Scale: {grasp_data['scale_name']}", "yellow")
            cprint(f"  Grasp: {grasp_data['grasp_idx']}/{grasp_data['num_grasps_in_file']}", "yellow")
            
            render_with_contact_map(grasp_data)
            
            # Brief pause between grasps (if not last one)
            if i < num_grasps - 1 and not headless:
                cprint(f"\nPress any key to continue to next grasp...", "green")
                if not viewer.viewer.closed:
                    # Just continue automatically
                    pass
    
    cprint(f"\n[INFO] Visualization completed!", "green")


def main(
    dexonomy_dir: str = "/home/guizhewei/guizhewei/Dexonomy_dataset",
    robots: Optional[List[RobotName]] = None,
    grasp_type: Optional[str] = None,
    object_id: Optional[str] = None,
    scale_name: str = "scale005",
    data_idx: int = 0,
    fps: int = 10,
    headless: bool = False,
    split: str = "train",
    num_grasps: int = 1,
    show_ground: bool = False,
    show_table: bool = False,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.vector,
    two_optimizers: bool = False,
    second_optimizer_type: str = "FINGERTIP",
    y_offset: float = 0.5,
    contact_map_path: Optional[str] = None,
    contact_map_idx: Optional[int] = None,
    show_contact_map: bool = False,
    show_object_points: bool = False,
    contact_sphere_radius: float = 0.003,
    contact_max_points: int = 5000,
    contact_sphere_alpha: float = 0.7,
    show_contact_map_parts: bool = False,
):
    """
    Visualize Dexonomy grasp dataset with optional contact map overlay
    
    Args:
        dexonomy_dir: Root directory of Dexonomy dataset
        robots: List of robot hands (e.g., shadow, allegro, omni)
        grasp_type: Specific grasp type (e.g., "1_Large_Diameter")
        object_id: Specific object ID (32-char hex string)
        scale_name: Object scale (scale005, scale008, etc.)
        data_idx: Sample index if object_id not specified
        fps: Rendering FPS
        headless: Save video instead of interactive viewing
        split: Dataset split (train/test/all)
        num_grasps: Number of consecutive grasps to visualize from data_idx (default: 1)
        show_ground: Show ground plane (default: False, cleaner view)
        show_table: Show table (default: False, cleaner view)
        hand_type: HandType (right or left)
        retargeting_type: RetargetingType for first optimizer (position, vector, fingertip, dexpilot)
        two_optimizers: Whether to use two optimizers (default: False)
        second_optimizer_type: Type of second optimizer ("VECTOR" or "FINGERTIP")
        y_offset: Y-axis spacing between robots for parallel display (default: 0.5m)
        contact_map_path: Path to contact map data file (.npy)
        contact_map_idx: Index in contact map data file
        show_contact_map: Whether to show contact map overlay (requires contact_map_path)
        show_object_points: Whether to show object surface points (requires contact_map_path)
        contact_sphere_radius: Radius of contact map spheres in meters (default: 0.008m=8mm)
        contact_max_points: Maximum number of contact sphere markers (default: 5000)
        contact_sphere_alpha: Transparency of contact spheres, 0-1 (default: 0.7)
    
    Examples:
        # View single grasp (default)
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 0
        
        # View with contact map overlay (default settings)
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_contact_map True
        
        # Sparse visualization - fewer points to see object mesh better
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_contact_map True \
            --contact_max_points 1000 \
            --contact_sphere_radius 0.005 \
            --contact_sphere_alpha 0.5
        
        # Dense visualization - more points for better coverage
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_contact_map True \
            --contact_max_points 8000 \
            --contact_sphere_radius 0.010
        
        # View 5 consecutive grasps
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 0 --num_grasps 5
        
        # Shadow Hand + OmniHand with vector retargeting
        python tools/contact_map/visualize_dexonomy_grasp.py --robots shadow_no_wrist omni --retargeting_type vector
        
        # Shadow Hand + OmniHand with two optimizers (vector + fingertip)
        python tools/contact_map/visualize_dexonomy_grasp.py --robots shadow_no_wrist omni --retargeting_type vector --two_optimizers True --second_optimizer_type FINGERTIP
        
        # Specific grasp type
        python tools/contact_map/visualize_dexonomy_grasp.py --grasp_type "1_Large_Diameter" --data_idx 0
        
        # Different objects/scales (browse by index)
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 100
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 500
        
        # Save video
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 0 --headless True
    """
    visualize_dexonomy_grasp(
        dexonomy_dir=dexonomy_dir,
        robots=robots,
        grasp_type=grasp_type,
        object_id=object_id,
        scale_name=scale_name,
        data_idx=data_idx,
        fps=fps,
        headless=headless,
        split=split,
        num_grasps=num_grasps,
        show_ground=show_ground,
        show_table=show_table,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
        two_optimizers=two_optimizers,
        second_optimizer_type=second_optimizer_type,
        y_offset=y_offset,
        contact_map_path=contact_map_path,
        contact_map_idx=contact_map_idx,
        show_contact_map=show_contact_map,
        show_object_points=show_object_points,
        contact_sphere_radius=contact_sphere_radius,
        contact_max_points=contact_max_points,
        contact_sphere_alpha=contact_sphere_alpha,
        show_contact_map_parts=show_contact_map_parts,
    )


if __name__ == "__main__":
    tyro.cli(main)

