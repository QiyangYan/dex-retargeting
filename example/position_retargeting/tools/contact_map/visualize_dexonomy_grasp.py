"""
Visualize Dexonomy grasp dataset with Shadow Hand and OmniHand in SAPIEN
支持叠加显示 contact map 可视化

Example usage:
    # Visualize Shadow Hand only
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset
    
    # Visualize Shadow Hand and OmniHand
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --robots shadow omni
    
    # Visualize with contact map overlay
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy --contact_map_idx 0
    
    # Visualize specific grasp type
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --grasp_type "1_Large_Diameter"
    
    # Visualize specific object
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --object_id "0004e882b42844eda8b4ab7379cb04c8"
"""
import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import tyro
from termcolor import cprint
import sapien
import torch

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
            return None
        
        sample = data[contact_map_idx]
        
        # 检查物体点云的 contact map
        if "contact_map_object" not in sample or len(sample["contact_map_object"]) == 0:
            cprint(f"[WARNING] No contact map (object) found for index {contact_map_idx}", "yellow")
            return None
        
        cprint(f"[INFO] Loaded contact map (object): shape={sample['contact_map_object'].shape}, mean={sample['contact_map_object'].mean():.4f}", "green")
        return sample
    except Exception as e:
        cprint(f"[ERROR] Failed to load contact map: {e}", "red")
        return None


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
    sphere_radius: float = 0.003,
    max_points: int = 5000,
):
    """
    在物体表面点上可视化 contact map（使用颜色编码的球体）
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        object_points: (M, 3) 物体表面点世界坐标
        contact_map: (M,) contact map 值数组，范围 [0, 1]
        sphere_radius: 球体半径（米）
        max_points: 最大显示点数（如果点数太多，进行采样）
    """
    if len(contact_map) != len(object_points):
        cprint(f"[WARNING] Contact map size ({len(contact_map)}) != object points size ({len(object_points)})", "yellow")
        # 使用较小的长度
        min_len = min(len(contact_map), len(object_points))
        contact_map = contact_map[:min_len]
        object_points = object_points[:min_len]
    
    # 如果点数太多，进行采样
    if len(object_points) > max_points:
        indices = np.random.choice(len(object_points), max_points, replace=False)
        object_points = object_points[indices]
        contact_map = contact_map[indices]
        cprint(f"[INFO] Sampled {max_points} points from {len(object_points)} total points", "white")
    
    # 清除现有的 contact map markers
    if not hasattr(viewer, 'object_contact_map_markers'):
        viewer.object_contact_map_markers = []
    else:
        for marker in viewer.object_contact_map_markers:
            viewer.scene.remove_actor(marker)
        viewer.object_contact_map_markers.clear()
    
    # 创建颜色映射：从蓝色（低接触）到红色（高接触）
    # contact_map 值范围 [0, 1]，映射到颜色
    colors = np.zeros((len(contact_map), 4))  # RGBA
    
    # 使用 colormap: 蓝色(0) -> 绿色(0.5) -> 红色(1)
    for i, cm_value in enumerate(contact_map):
        if cm_value < 0.5:
            # 蓝色到绿色
            r = 0.0
            g = cm_value * 2.0
            b = 1.0 - cm_value * 2.0
        else:
            # 绿色到红色
            r = (cm_value - 0.5) * 2.0
            g = 1.0 - (cm_value - 0.5) * 2.0
            b = 0.0
        
        colors[i] = [r, g, b, 0.8]  # RGBA，alpha=0.8
    
    # 创建球体 markers
    cprint(f"[INFO] Creating {len(object_points)} object contact map markers...", "cyan")
    
    for i, (pos, color) in enumerate(zip(object_points, colors)):
        # 创建球体 material
        sphere_material = sapien.render.RenderMaterial()
        sphere_material.set_base_color(color)
        sphere_material.set_roughness(0.3)
        sphere_material.set_metallic(0.1)
        sphere_material.set_specular(0.8)
        
        # 创建球体
        builder = viewer.scene.create_actor_builder()
        builder.add_sphere_visual(radius=sphere_radius, material=sphere_material)
        marker = builder.build_static(name=f"object_contact_map_marker_{i}")
        marker.set_pose(sapien.Pose(pos))
        viewer.object_contact_map_markers.append(marker)
    
    cprint(f"[INFO] Created {len(viewer.object_contact_map_markers)} object contact map markers", "green")
    cprint(f"  Contact map range: [{contact_map.min():.4f}, {contact_map.max():.4f}]", "white")
    cprint(f"  Contact map mean: {contact_map.mean():.4f}", "white")
    cprint(f"  Strong contacts (>0.5): {(contact_map > 0.5).sum()}", "white")


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
    
    # Create viewer
    viewer = DexonomyGraspSAPIENViewer(
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
    
    # Helper function to render with contact map and/or object points overlay
    def render_with_contact_map(grasp_data_item):
        """渲染单个抓取，如果启用了 contact map 或 object points 则叠加显示"""
        # 如果启用了 contact map 或 object points，先设置手部姿态
        if (show_contact_map or show_object_points) and contact_map_data is not None:
            try:
                # 加载物体
                viewer.load_object(grasp_data_item)
                
                # 设置 Shadow Hand 姿态
                grasp_qpos = grasp_data_item["grasp_qpos"]
                viewer.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
                
                # 可视化物体点云上的 contact map
                if show_contact_map:
                    if "object_point_cloud" in contact_map_data and "contact_map_object" in contact_map_data:
                        object_points = contact_map_data["object_point_cloud"]
                        contact_map_object = contact_map_data["contact_map_object"]
                        
                        if len(object_points) > 0 and len(contact_map_object) > 0:
                            # 使用 contact map 值来着色物体点云
                            visualize_object_surface_points_with_contact_map(
                                viewer,
                                object_points,
                                contact_map_object,
                                sphere_radius=0.003,
                                max_points=5000,
                            )
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
        # Render single grasp
        render_with_contact_map(grasp_data)
    else:
        # Render multiple consecutive grasps
        cprint(f"\n[INFO] Rendering {num_grasps} consecutive grasps starting from index {data_idx}", "cyan")
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
    
    Examples:
        # View single grasp (default)
        python tools/contact_map/visualize_dexonomy_grasp.py --data_idx 0
        
        # View with contact map overlay
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_contact_map True
        
        # View with object surface points
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_object_points True
        
        # View with both contact map and object points
        python tools/contact_map/visualize_dexonomy_grasp.py \
            --data_idx 0 \
            --contact_map_path /path/to/grasp_poses_retargeted_xxx.npy \
            --contact_map_idx 0 \
            --show_contact_map True \
            --show_object_points True
        
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
    )


if __name__ == "__main__":
    tyro.cli(main)

