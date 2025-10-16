from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from sklearn.base import defaultdict
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer
from hand_robot_viewer_img import RobotHandDatasetSAPIENViewer_IMG
from contact_detection import detect_contact_combined, load_object_mesh, analyze_contact_results
from contact_detection_sapien import detect_contact_in_viewer, analyze_contact_results as analyze_sapien_results, get_contact_summary
import copy

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_

actuated_dof_indices = []

"""
Indices Info:
0-6: wrist pose
7: index, range: [0, 1.47]
8: middle, range: [0, 1.47]
9: ring, range: [0, 1.47]
10: pinky, range: [0, 1.47]
11: thumb base, range: [0, 0.6]
12: thumb, range: [0, 1.308]


Isaac range:


"""


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int, img: bool = False, retargeting_type: str = "POSITION", save_grasp_pose: bool = False, data_id: Optional[int] = None, two_optimizers: bool = False, second_optimizer_type: str = "VECTOR", save_contact_info: bool = False, use_sapien_contact: bool = True, visualize: bool = True):
    # Determine headless mode
    if save_grasp_pose:
        headless = True
    elif save_contact_info and not visualize:
        headless = True  # Only headless if saving contact and not visualizing
    else:
        headless = False

    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=False)
    elif img:
        if retargeting_type not in ["POSITION", "VECTOR", "FINGERTIP", "DEXPILOT"]:
            raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
        if retargeting_type == "POSITION":
            retargeting_type = RetargetingType.position
        elif retargeting_type == "VECTOR":
            retargeting_type = RetargetingType.vector
        elif retargeting_type == "FINGERTIP":
            retargeting_type = RetargetingType.fingertip
        elif retargeting_type == "DEXPILOT":
            retargeting_type = RetargetingType.dexpilot
        viewer = RobotHandDatasetSAPIENViewer_IMG(
            list(robots), HandType.right, headless=headless, retargeting_type=retargeting_type
        )
    else:
        if retargeting_type not in ["POSITION", "VECTOR", "FINGERTIP", "DEXPILOT"]:
            raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
        if retargeting_type == "POSITION":
            retargeting_type = RetargetingType.position
        elif retargeting_type == "VECTOR":
            retargeting_type = RetargetingType.vector
        elif retargeting_type == "FINGERTIP":
            retargeting_type = RetargetingType.fingertip
        elif retargeting_type == "DEXPILOT":
            retargeting_type = RetargetingType.dexpilot
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=headless, retargeting_type=retargeting_type, two_optimizers=two_optimizers, second_optimizer_type=second_optimizer_type
        )
    
    grasp_pose_dict = defaultdict(dict)
    
    # If data_id is not specified, process all data
    if data_id is None:
        data_indices = range(len(dataset))
        print(f"Processing all {len(dataset)} data entries...")
    else:
        data_indices = [data_id]
        print(f"Processing data entry {data_id}...")
    
    for i, sampled_data in enumerate(dataset):
        if i not in data_indices:
            continue
        
        if True:  # Replace the "if i == data_id:" condition
            for key, value in sampled_data.items():
                if "pose" not in key:
                    print(f"{key}: {value}")
            data = copy.deepcopy(sampled_data)   # <- important
            if not img:
                viewer.reset_env()                   # <- add (see below)
            viewer.load_object_hand(data)
            
            # Render based on flags
            if save_grasp_pose or save_contact_info: 
                grasp_pose = viewer.render_dexycb_data(sampled_data, fps)
                grasp_pose_dict[i] = grasp_pose
                
                # Add contact information if requested
                if save_contact_info:
                    if use_sapien_contact:
                        contact_info = detect_contact_sapien(viewer, sampled_data, data_root)
                    else:
                        contact_info = detect_contact_for_last_frame(sampled_data, data_root)
                    
                    # Only add the 21-dim contact labels array to the grasp_pose_dict
                    if "error" not in contact_info:
                        grasp_pose_dict[i]["contact_labels"] = contact_info["contact_labels"]
                        print(f"Contact detection completed for data {i}")
                    else:
                        print(f"Contact detection failed for data {i}: {contact_info['error']}")
            elif visualize:
                # Visualize if requested
                viewer.render_dexycb_data(sampled_data, fps)

    # save dict in npy format with both grasp poses and contact labels (if requested)
    if save_grasp_pose or save_contact_info:
        name = input("Enter the name for the file (without extension): ")
        save_path = data_root / f"grasp_poses_{name}.npy"
        np.save(save_path, dict(grasp_pose_dict))
        if save_contact_info:
            print(f"Grasp poses with contact labels saved to {save_path}")
        else:
            print(f"Grasp poses saved to {save_path}")


def detect_contact_sapien(viewer, sampled_data: dict, data_root: Path) -> dict:
    """
    Detect contact using SAPIEN physics engine.
    
    Args:
        viewer: SAPIEN viewer with scene and loaded objects
        sampled_data: Data from DexYCB dataset
        data_root: Root directory of DexYCB dataset
    
    Returns:
        Dictionary containing contact information
    """
    try:
        # Get the last frame data
        capture_name = sampled_data["capture_name"]
        hand_pose = sampled_data["hand_pose"]
        object_pose = sampled_data["object_pose"]
        ycb_ids = sampled_data["ycb_ids"]
        
        # Find the target object (the one being manipulated)
        # Use ycb_grasp_ind from meta.yml to identify which object is being grasped
        capture_dir = data_root / "20200709-subject-01" / capture_name
        meta_file = capture_dir / "meta.yml"
        
        import yaml
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        
        ycb_grasp_ind = meta.get('ycb_grasp_ind', 0)  # Default to 0 if not found
        target_object_idx = ycb_grasp_ind
        
        # Validate index
        if target_object_idx >= len(ycb_ids):
            print(f"Warning: ycb_grasp_ind ({target_object_idx}) >= len(ycb_ids) ({len(ycb_ids)}), using 0")
            target_object_idx = 0
        
        target_object_id = ycb_ids[target_object_idx]
        
        # Load labels for the last frame to get joint positions AND object pose
        capture_dir = data_root / "20200709-subject-01" / capture_name
        camera_dirs = [d for d in capture_dir.iterdir() if d.is_dir() and d.name.startswith("8")]
        
        if not camera_dirs:
            print(f"Warning: No camera directories found in {capture_dir}")
            return {"error": "Camera directories not found"}
        
        # Use the first camera directory
        camera_dir = camera_dirs[0]
        
        # Find the last frame labels file
        label_files = list(camera_dir.glob("labels_*.npz"))
        if not label_files:
            print(f"Warning: No label files found in {camera_dir}")
            return {"error": "Label files not found"}
        
        # Sort by frame number and get the last one
        label_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        last_label_file = label_files[-1]
        
        # Load labels (contains both joint_3d and pose_y in camera frame!)
        labels_data = np.load(last_label_file)
        hand_joints_3d = labels_data["joint_3d"][0]  # Shape: (21, 3) - in camera frame
        object_pose_camera = labels_data["pose_y"][target_object_idx]  # Shape: (3, 4) - [R|t] in camera frame
        
        # Extract object position from pose_y (the translation vector)
        object_position_camera = object_pose_camera[:, 3]  # Shape: (3,)
        
        # Create 7D pose format [x, y, z, qx, qy, qz, qw] for compatibility
        # For contact detection, we mainly need the position
        from scipy.spatial.transform import Rotation
        rotation_matrix = object_pose_camera[:, :3]
        quat = Rotation.from_matrix(rotation_matrix).as_quat()  # [qx, qy, qz, qw]
        last_object_pose = np.concatenate([object_position_camera, quat])  # Shape: (7,)
        
        # Check for invalid joints
        valid_mask = ~np.all(hand_joints_3d == -1, axis=1)
        if not valid_mask.any():
            print("Warning: No valid hand joints detected")
            return {"error": "No valid hand joints"}
        
        # Use SAPIEN contact detection
        print(f"\nDetecting contact using SAPIEN for {capture_name}...")
        print(f"Grasp object index: {target_object_idx} (ycb_grasp_ind from meta.yml)")
        print(f"Target object: YCB ID {target_object_id} (Object Index {target_object_idx} in scene)")
        print(f"Scene objects: {ycb_ids} (total {len(ycb_ids)} objects)")
        print(f"Hand joints 3D shape: {hand_joints_3d.shape}")
        print(f"Valid joints: {valid_mask.sum()}/21")
        print(f"Sample joint positions: {hand_joints_3d[:3]}")
        print(f"Last object pose: {last_object_pose}")
        
        # Get models directory
        models_dir = data_root / "models"
        
        contact_results = detect_contact_in_viewer(
            viewer,
            hand_joints_3d,
            last_object_pose,  # Now in camera frame, same as hand_joints_3d!
            object_id=target_object_id,  # Pass object ID for mesh loading
            models_dir=models_dir,        # Pass models directory
            object_idx=target_object_idx,
            sphere_radius=0.01,
            distance_threshold=0.02  # 1cm threshold for surface distance (mesh-based detection)
        )
        
        # Prepare contact info with 21-dim contact array as primary data
        contact_info = {
            # Primary data: 21-dimensional 0/1 contact array (corresponds to MANO joint indices)
            "contact_labels": contact_results['contact_labels'],  # Shape: (21,), dtype: int32
            
            # Metadata
            "capture_name": capture_name,
            "target_object_id": target_object_id,
            "target_object_idx": target_object_idx,
            "last_frame": len(hand_pose) - 1,
            "detection_method": "SAPIEN",
            
            # Additional info (optional, can be removed if not needed)
            "distances_to_center": contact_results.get('distances_to_center'),  # Shape: (21,)
            "sphere_radius": contact_results.get('sphere_radius'),
            "distance_threshold": contact_results.get('distance_threshold'),
        }
        
        # Add summary statistics
        summary = get_contact_summary(contact_results['contact_labels'])
        contact_info.update(summary)
        
        # Print analysis
        print(f"\nSAPIEN contact detection for {capture_name}, object {target_object_id}:")
        print(f"Contact labels (21-dim array): {contact_results['contact_labels']}")
        analyze_sapien_results(contact_results)
        
        return contact_info
        
    except Exception as e:
        print(f"Error in SAPIEN contact detection: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def detect_contact_for_last_frame(sampled_data: dict, data_root: Path) -> dict:
    """
    Detect contact information for the last frame of the sequence.
    
    Args:
        sampled_data: Data from DexYCB dataset
        data_root: Root directory of DexYCB dataset
    
    Returns:
        Dictionary containing contact information
    """
    try:
        # Get the last frame data
        capture_name = sampled_data["capture_name"]
        hand_pose = sampled_data["hand_pose"]
        object_pose = sampled_data["object_pose"]
        ycb_ids = sampled_data["ycb_ids"]
        
        # Find the target object (the one being manipulated)
        # Assume the first object is the target for now
        target_object_idx = 0
        target_object_id = ycb_ids[target_object_idx]
        
        # Get last frame poses
        last_hand_pose = hand_pose[-1]  # Shape: (1, 51)
        last_object_pose = object_pose[-1, target_object_idx]  # Shape: (7,)
        
        # Load object mesh
        models_dir = data_root / "models"
        object_mesh = load_object_mesh(target_object_id, models_dir)
        
        if object_mesh is None:
            print(f"Warning: Could not load mesh for object {target_object_id}")
            return {"error": "Mesh not found"}
        
        # Load labels for the last frame to get joint positions
        capture_dir = data_root / "20200709-subject-01" / capture_name
        camera_dirs = [d for d in capture_dir.iterdir() if d.is_dir() and d.name.startswith("8")]
        
        if not camera_dirs:
            print(f"Warning: No camera directories found in {capture_dir}")
            return {"error": "Camera directories not found"}
        
        # Use the first camera directory
        camera_dir = camera_dirs[0]
        
        # Find the last frame labels file
        label_files = list(camera_dir.glob("labels_*.npz"))
        if not label_files:
            print(f"Warning: No label files found in {camera_dir}")
            return {"error": "Label files not found"}
        
        # Sort by frame number and get the last one
        label_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        last_label_file = label_files[-1]
        
        # Load labels
        labels_data = np.load(last_label_file)
        hand_joints_3d = labels_data["joint_3d"][0]  # Shape: (21, 3)
        hand_joints_2d = labels_data["joint_2d"][0]  # Shape: (21, 2)
        segmentation = labels_data["seg"]  # Shape: (H, W)
        
        # Determine object segmentation ID (assuming it's the largest non-zero value)
        unique_seg_ids = np.unique(segmentation)
        object_seg_ids = unique_seg_ids[unique_seg_ids > 0]
        if len(object_seg_ids) > 0:
            object_seg_id = object_seg_ids[0]  # Use the first non-zero ID
        else:
            object_seg_id = 1  # Default fallback
        
        # Detect contact
        contact_results = detect_contact_combined(
            hand_joints_3d=hand_joints_3d,
            hand_joints_2d=hand_joints_2d,
            object_pose=last_object_pose,
            object_mesh=object_mesh,
            segmentation=segmentation,
            object_seg_id=object_seg_id,
            contact_threshold=0.02
        )
        
        # Prepare contact info with 21-dim contact array as primary data
        contact_info = {
            # Primary data: 21-dimensional 0/1 contact array (corresponds to MANO joint indices)
            "contact_labels": contact_results['combined_contact'],  # Shape: (21,), dtype: int32
            
            # Metadata
            "capture_name": capture_name,
            "target_object_id": target_object_id,
            "target_object_idx": target_object_idx,
            "last_frame": len(hand_pose) - 1,
            "detection_method": "Geometric",
            "object_seg_id": object_seg_id,
            "contact_threshold": 0.02,
            
            # Additional detection results (optional)
            "distance_contact": contact_results['distance_contact'],      # Shape: (21,)
            "segmentation_contact": contact_results['segmentation_contact'],  # Shape: (21,)
            "min_distances": contact_results['min_distances'],            # Shape: (21,)
        }
        
        # Print analysis
        print(f"\nGeometric contact detection for {capture_name}, object {target_object_id}:")
        print(f"Contact labels (21-dim array): {contact_results['combined_contact']}")
        analyze_contact_results(contact_results)
        
        return contact_info
        
    except Exception as e:
        print(f"Error in contact detection: {e}")
        return {"error": str(e)}


def main(dexycb_dir: str="/home/guizhewei/guizhewei/Dexycb_dataset", robots: Optional[List[RobotName]] = None, fps: int = 10, img: bool = False, retargeting_type: str = "POSITION", save_grasp_pose: bool = False, data_id: Optional[int] = None, two_optimizers: bool = False, second_optimizer_type: str = "VECTOR", save_contact_info: bool = False, use_sapien_contact: bool = True, visualize: bool = True):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory
        img: whether to use single frame for retargeting
        retargeting_type: retargeting type, either "POSITION", "VECTOR", "FINGERTIP", or "DEXPILOT"
        save_grasp_pose: whether to save grasp poses to file
        data_id: which data to process (None = process all data, specific int = process only that index)
        two_optimizers: whether to use two optimizers for retargeting
        second_optimizer_type: type of the second optimizer when two_optimizers=True, either "VECTOR", "FINGERTIP", or "DEXPILOT"
        save_contact_info: whether to save contact information for MANO joints
        use_sapien_contact: whether to use SAPIEN physics-based contact detection (True) or geometric method (False)
        visualize: whether to show visualization window (True) or run headless (False)

    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    print(robot_dir)
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    viz_hand_object(robots, data_root, fps, img, retargeting_type, save_grasp_pose, data_id, two_optimizers, second_optimizer_type, save_contact_info, use_sapien_contact, visualize)


if __name__ == "__main__":
    tyro.cli(main)
