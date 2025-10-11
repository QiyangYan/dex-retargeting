"""
Visualize contact detection results with 3D rendering.
Shows hand joints and object mesh with contact joints highlighted.
"""

import sys
sys.path.insert(0, '/home/guizhewei/guizhewei/dex-ycb-toolkit')

import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Optional
import argparse

from contact_detection_sapien import load_object_mesh, YCB_CLASSES


def get_mano_joint_names():
    """Get MANO joint names."""
    return [
        "wrist",
        "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "little_mcp", "little_pip", "little_dip", "little_tip",
    ]


def visualize_contact_3d(
    hand_joints_3d: np.ndarray,
    contact_labels: np.ndarray,
    object_pose: np.ndarray,
    object_id: int,
    models_dir: Path,
    distances: Optional[np.ndarray] = None,
    show_distances: bool = True,
    save_image: Optional[str] = None
):
    """
    Visualize hand joints and object with contact highlighting.
    
    Args:
        hand_joints_3d: Hand joint positions (21, 3)
        contact_labels: Binary contact labels (21,)
        object_pose: Object pose [x, y, z, qx, qy, qz, qw] (7,)
        object_id: YCB object ID
        models_dir: Path to models directory
        distances: Optional distances from joints to surface (21,)
        show_distances: Whether to show distance labels
        save_image: Optional path to save screenshot
    """
    print(f"\n{'='*60}")
    print("🎨 Visualizing Contact Detection Results")
    print(f"{'='*60}\n")
    
    # Load and transform object mesh
    print(f"Loading mesh for object ID {object_id} ({YCB_CLASSES.get(object_id, 'Unknown')})...")
    mesh = load_object_mesh(object_id, models_dir)
    
    if mesh is None:
        print("ERROR: Failed to load mesh!")
        return
    
    # Transform mesh to world frame
    object_position = object_pose[:3]
    object_quat = object_pose[3:]
    rotation = Rotation.from_quat(object_quat).as_matrix()
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = object_position
    
    mesh_transformed = mesh.copy()
    mesh_transformed.apply_transform(transform)
    
    # Set mesh color (light gray)
    mesh_transformed.visual.face_colors = [200, 200, 200, 100]
    
    # Create joint visualization
    joint_names = get_mano_joint_names()
    
    # Separate contact and non-contact joints
    contact_joints = []
    non_contact_joints = []
    
    for i, (pos, label) in enumerate(zip(hand_joints_3d, contact_labels)):
        if np.all(pos == -1):
            continue  # Skip invalid joints
        
        if label == 1:
            contact_joints.append(pos)
        else:
            non_contact_joints.append(pos)
    
    contact_joints = np.array(contact_joints) if contact_joints else np.zeros((0, 3))
    non_contact_joints = np.array(non_contact_joints) if non_contact_joints else np.zeros((0, 3))
    
    # Create point clouds
    geometries = [mesh_transformed]
    
    # Non-contact joints (green)
    if len(non_contact_joints) > 0:
        non_contact_cloud = trimesh.PointCloud(
            non_contact_joints,
            colors=[0, 255, 0, 255]  # Green
        )
        geometries.append(non_contact_cloud)
    
    # Contact joints (red, larger)
    if len(contact_joints) > 0:
        contact_cloud = trimesh.PointCloud(
            contact_joints,
            colors=[255, 0, 0, 255]  # Red
        )
        geometries.append(contact_cloud)
    
    # Add spheres for contact joints to make them more visible
    for i, (pos, label) in enumerate(zip(hand_joints_3d, contact_labels)):
        if np.all(pos == -1) or label == 0:
            continue
        
        # Create a small sphere at contact joint
        sphere = trimesh.primitives.Sphere(radius=0.005, center=pos)
        sphere.visual.face_colors = [255, 0, 0, 200]  # Red
        geometries.append(sphere)
    
    # Add lines connecting joints (skeleton)
    # MANO hand topology
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Little
    ]
    
    for start_idx, end_idx in connections:
        start_pos = hand_joints_3d[start_idx]
        end_pos = hand_joints_3d[end_idx]
        
        if np.all(start_pos == -1) or np.all(end_pos == -1):
            continue
        
        # Create line as cylinder for better visibility
        # Calculate direction and length
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            continue
        
        # Create thin cylinder
        cylinder = trimesh.creation.cylinder(
            radius=0.001,  # 1mm radius
            height=length,
            sections=8
        )
        
        # Color based on contact
        if contact_labels[start_idx] or contact_labels[end_idx]:
            cylinder.visual.face_colors = [255, 0, 0, 200]  # Red
        else:
            cylinder.visual.face_colors = [0, 255, 0, 150]  # Green
        
        # Transform cylinder to correct position and orientation
        # Default cylinder is along z-axis, need to rotate to align with direction
        direction_normalized = direction / length
        z_axis = np.array([0, 0, 1])
        
        # Rotation from z-axis to direction
        if np.allclose(direction_normalized, z_axis):
            rotation_matrix = np.eye(3)
        elif np.allclose(direction_normalized, -z_axis):
            rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            axis = np.cross(z_axis, direction_normalized)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z_axis, direction_normalized))
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
        
        # Transform
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = (start_pos + end_pos) / 2  # Center
        
        cylinder.apply_transform(transform)
        geometries.append(cylinder)
    
    # Create scene
    scene = trimesh.Scene(geometries)
    
    # Print summary
    num_contacts = contact_labels.sum()
    print(f"\n📊 Contact Summary:")
    print(f"  Total contacts: {num_contacts}/21")
    print(f"  Non-contacts: {21 - num_contacts}/21")
    
    if num_contacts > 0:
        print(f"\n✅ Contact joints:")
        for i, (label, name) in enumerate(zip(contact_labels, joint_names)):
            if label == 1:
                dist_str = f" ({distances[i]*1000:.2f}mm)" if distances is not None else ""
                print(f"    [{i:2d}] {name:12s}{dist_str}")
    
    print(f"\n🎨 Visualization Legend:")
    print(f"  🔴 Red spheres = Contact joints")
    print(f"  🟢 Green points = Non-contact joints")
    print(f"  ⚪ Gray mesh = Object")
    print(f"  Lines = Hand skeleton")
    
    print(f"\n{'='*60}")
    print("Opening 3D viewer... (close window to continue)")
    print(f"{'='*60}\n")
    
    # Show visualization
    if save_image:
        # Save screenshot
        try:
            # Try with resolution parameter
            png = scene.save_image(resolution=[1920, 1080])
            with open(save_image, 'wb') as f:
                f.write(png)
            print(f"Screenshot saved to: {save_image}")
        except Exception as e:
            print(f"Warning: Could not save with resolution, using default: {e}")
            # Fallback: save without resolution
            png = scene.save_image()
            with open(save_image, 'wb') as f:
                f.write(png)
            print(f"Screenshot saved to: {save_image}")
    else:
        # Interactive viewer
        scene.show()


def visualize_from_saved_data(
    contact_file: str,
    data_id: int = 0,
    dexycb_dir: str = "/home/guizhewei/guizhewei/Dexycb_dataset",
    save_image: Optional[str] = None
):
    """
    Load saved contact info and visualize it.
    
    Args:
        contact_file: Path to saved .npy file (grasp_poses with contact_labels or standalone contact_info)
        data_id: Which data ID to visualize
        dexycb_dir: Path to DexYCB dataset
        save_image: Optional path to save screenshot
    """
    from dataset import DexYCBVideoDataset
    import yaml
    
    # Load data file
    saved_data = np.load(contact_file, allow_pickle=True).item()
    
    if data_id not in saved_data:
        print(f"ERROR: Data ID {data_id} not found in {contact_file}")
        print(f"Available IDs: {list(saved_data.keys())}")
        return
    
    data_entry = saved_data[data_id]
    
    # Check if this is new format (grasp_poses with contact_labels) or old format (standalone contact_info)
    # New format: data_entry is a dict with 'contact_labels' key
    # Old format: data_entry is a dict with 'target_object_idx' and 'target_object_id' keys
    
    if 'contact_labels' in data_entry:
        contact_labels = data_entry['contact_labels']
        
        # Check if it's old format with metadata or new format without metadata
        if 'target_object_idx' in data_entry and 'target_object_id' in data_entry:
            # Old standalone contact_info format
            target_object_idx = data_entry['target_object_idx']
            target_object_id = data_entry['target_object_id']
            print("📂 Detected old format (standalone contact_info)")
        else:
            # New format: integrated in grasp_poses, need to get metadata from dataset
            print("📂 Detected new format (contact_labels in grasp_poses)")
            target_object_idx = None
            target_object_id = None
    else:
        print(f"ERROR: No 'contact_labels' found in data entry {data_id}")
        print(f"Available keys: {list(data_entry.keys())}")
        return
    
    # Load dataset
    data_root = Path(dexycb_dir)
    models_dir = data_root / "models"
    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    sampled_data = dataset[data_id]
    
    # Get capture name and ycb_ids
    capture_name = sampled_data["capture_name"]
    ycb_ids = sampled_data["ycb_ids"]
    
    # If target_object info not in file, get it from meta.yml
    if target_object_idx is None or target_object_id is None:
        capture_dir = data_root / "20200709-subject-01" / capture_name
        meta_file = capture_dir / "meta.yml"
        
        with open(meta_file, 'r') as f:
            meta = yaml.safe_load(f)
        
        target_object_idx = meta.get('ycb_grasp_ind', 0)
        
        # Validate index
        if target_object_idx >= len(ycb_ids):
            print(f"Warning: ycb_grasp_ind ({target_object_idx}) >= len(ycb_ids) ({len(ycb_ids)}), using 0")
            target_object_idx = 0
        
        target_object_id = ycb_ids[target_object_idx]
        print(f"Retrieved from meta.yml: target_object_idx={target_object_idx}, target_object_id={target_object_id}")
    
    # Load labels
    capture_dir = data_root / "20200709-subject-01" / capture_name
    camera_dirs = [d for d in capture_dir.iterdir() if d.is_dir() and d.name.startswith("8")]
    
    if not camera_dirs:
        print(f"ERROR: No camera directories found in {capture_dir}")
        return
    
    camera_dir = camera_dirs[0]
    
    label_files = list(camera_dir.glob("labels_*.npz"))
    if not label_files:
        print(f"ERROR: No label files found in {camera_dir}")
        return
    
    label_files.sort(key=lambda x: int(x.stem.split('_')[1]))
    last_label_file = label_files[-1]
    
    labels_data = np.load(last_label_file)
    hand_joints_3d = labels_data["joint_3d"][0]
    object_pose_camera = labels_data["pose_y"][target_object_idx]
    
    # Convert to 7D pose
    object_position = object_pose_camera[:, 3]
    rotation = object_pose_camera[:, :3]
    quat = Rotation.from_matrix(rotation).as_quat()
    object_pose_7d = np.concatenate([object_position, quat])
    
    # Visualize
    print(f"\nLoading data {data_id}: {capture_name}")
    print(f"Object: {YCB_CLASSES.get(target_object_id, 'Unknown')} (ID {target_object_id})")
    
    visualize_contact_3d(
        hand_joints_3d=hand_joints_3d,
        contact_labels=contact_labels,
        object_pose=object_pose_7d,
        object_id=target_object_id,
        models_dir=models_dir,
        distances=data_entry.get('distances_to_center'),
        save_image=save_image
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize contact detection results")
    parser.add_argument('contact_file', type=str, help='Path to saved contact info .npy file')
    parser.add_argument('--data-id', type=int, default=0, help='Data ID to visualize')
    parser.add_argument('--dexycb-dir', type=str, 
                       default='/home/guizhewei/guizhewei/Dexycb_dataset',
                       help='Path to DexYCB dataset')
    parser.add_argument('--save-image', type=str, default=None,
                       help='Save screenshot to this path (e.g., contact_vis.png)')
    
    args = parser.parse_args()
    
    visualize_from_saved_data(
        contact_file=args.contact_file,
        data_id=args.data_id,
        dexycb_dir=args.dexycb_dir,
        save_image=args.save_image
    )


if __name__ == "__main__":
    main()

