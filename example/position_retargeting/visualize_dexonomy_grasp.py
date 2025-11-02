"""
Visualize Dexonomy grasp dataset with Shadow Hand and OmniHand in SAPIEN

Example usage:
    # Visualize Shadow Hand only
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset
    
    # Visualize Shadow Hand and OmniHand
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --robots shadow omni
    
    # Visualize specific grasp type
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --grasp_type "1_Large_Diameter"
    
    # Visualize specific object
    python visualize_dexonomy_grasp.py --dexonomy_dir /path/to/Dexonomy_dataset --object_id "0004e882b42844eda8b4ab7379cb04c8"
"""
from pathlib import Path
from typing import Optional, List

import numpy as np
import tyro
from termcolor import cprint

from dexonomy_dataset import DexonomyGraspDataset
from dexonomy_viewer import DexonomyGraspSAPIENViewer
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


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
):
    """
    Visualize Dexonomy grasp dataset with Shadow Hand and optional retargeting to other hands
    
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
    """
    # Setup paths
    data_root = Path(dexonomy_dir).absolute()
    if not data_root.exists():
        raise ValueError(f"Dexonomy directory does not exist: {data_root}")
    
    robot_dir = Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
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
    
    # Render grasp(s)
    if num_grasps == 1:
        # Render single grasp
        viewer.render_grasp_single(grasp_data, fps=fps, y_offset=y_offset)
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
            
            viewer.render_grasp_single(grasp_data, fps=fps, y_offset=y_offset)
            
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
):
    """
    Visualize Dexonomy grasp dataset
    
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
    
    Examples:
        # View single grasp (default)
        python visualize_dexonomy_grasp.py --data_idx 0
        
        # View 5 consecutive grasps
        python visualize_dexonomy_grasp.py --data_idx 0 --num_grasps 5
        
        # Shadow Hand + OmniHand with vector retargeting
        python visualize_dexonomy_grasp.py --robots shadow_no_wrist omni --retargeting_type vector
        
        # Shadow Hand + OmniHand with two optimizers (vector + fingertip)
        python visualize_dexonomy_grasp.py --robots shadow_no_wrist omni --retargeting_type vector --two_optimizers True --second_optimizer_type FINGERTIP
        
        # Specific grasp type
        python visualize_dexonomy_grasp.py --grasp_type "1_Large_Diameter" --data_idx 0
        
        # Different objects/scales (browse by index)
        python visualize_dexonomy_grasp.py --data_idx 100
        python visualize_dexonomy_grasp.py --data_idx 500
        
        # Save video
        python visualize_dexonomy_grasp.py --data_idx 0 --headless True
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
    )


if __name__ == "__main__":
    tyro.cli(main)

