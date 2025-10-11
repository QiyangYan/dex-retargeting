"""
Contact detection utilities for DexYCB dataset.
Detects contact between MANO hand joints and objects.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import trimesh
from scipy.spatial.distance import cdist


def load_object_mesh(object_id: int, models_dir: Path) -> Optional[trimesh.Trimesh]:
    """Load object mesh from DexYCB models directory."""
    # YCB object ID to name mapping
    YCB_CLASSES = {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }
    
    try:
        if object_id not in YCB_CLASSES:
            print(f"Unknown object ID: {object_id}")
            return None
            
        object_name = YCB_CLASSES[object_id]
        mesh_file = models_dir / object_name / "textured_simple.obj"
        
        if mesh_file.exists():
            mesh = trimesh.load(str(mesh_file))
            return mesh
        else:
            print(f"Mesh file not found: {mesh_file}")
            return None
    except Exception as e:
        print(f"Error loading mesh for object {object_id}: {e}")
        return None


def detect_contact_distance(
    hand_joints: np.ndarray,  # Shape: (21, 3)
    object_pose: np.ndarray,  # Shape: (7,) - [x, y, z, qx, qy, qz, qw]
    object_mesh: trimesh.Trimesh,
    contact_threshold: float = 0.02
) -> np.ndarray:
    """
    Detect contact based on distance between hand joints and object surface.
    
    Args:
        hand_joints: MANO hand joint positions in world coordinates
        object_pose: Object pose [x, y, z, qx, qy, qz, qw]
        object_mesh: Object mesh
        contact_threshold: Distance threshold for contact detection (meters)
    
    Returns:
        contact_labels: Binary array of shape (21,) indicating contact for each joint
    """
    # Transform object mesh to world coordinates
    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(object_pose[3:7])  # [qx, qy, qz, qw]
    rotation_matrix = rotation.as_matrix()
    
    # Create transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = object_pose[:3]
    
    # Transform mesh to world coordinates
    object_mesh_world = object_mesh.copy()
    object_mesh_world.apply_transform(transform_matrix)
    
    # Get mesh vertices
    mesh_vertices = object_mesh_world.vertices
    
    # Calculate distances from each hand joint to mesh vertices
    distances = cdist(hand_joints, mesh_vertices)
    
    # Find minimum distance for each joint
    min_distances = np.min(distances, axis=1)
    
    # Determine contact based on threshold
    contact_labels = (min_distances <= contact_threshold).astype(np.int32)
    
    return contact_labels


def detect_contact_segmentation(
    hand_joints_2d: np.ndarray,  # Shape: (21, 2)
    segmentation: np.ndarray,    # Shape: (H, W)
    object_seg_id: int
) -> np.ndarray:
    """
    Detect contact based on segmentation mask.
    
    Args:
        hand_joints_2d: Hand joint 2D coordinates in image space
        segmentation: Segmentation mask
        object_seg_id: Segmentation ID for the target object
    
    Returns:
        contact_labels: Binary array of shape (21,) indicating contact for each joint
    """
    contact_labels = np.zeros(21, dtype=np.int32)
    
    h, w = segmentation.shape
    
    for i, joint_2d in enumerate(hand_joints_2d):
        x, y = joint_2d.astype(int)
        
        # Check if joint is within image bounds
        if 0 <= x < w and 0 <= y < h:
            # Check if joint is on the object segmentation
            if segmentation[y, x] == object_seg_id:
                contact_labels[i] = 1
                
                # Also check a small neighborhood around the joint
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if segmentation[ny, nx] == object_seg_id:
                                contact_labels[i] = 1
                                break
                    if contact_labels[i] == 1:
                        break
    
    return contact_labels


def detect_contact_combined(
    hand_joints_3d: np.ndarray,  # Shape: (21, 3)
    hand_joints_2d: np.ndarray,  # Shape: (21, 2)
    object_pose: np.ndarray,      # Shape: (7,)
    object_mesh: trimesh.Trimesh,
    segmentation: np.ndarray,     # Shape: (H, W)
    object_seg_id: int,
    contact_threshold: float = 0.02
) -> Dict[str, np.ndarray]:
    """
    Combined contact detection using both distance and segmentation methods.
    
    Returns:
        Dictionary containing:
        - 'distance_contact': Contact labels from distance method
        - 'segmentation_contact': Contact labels from segmentation method
        - 'combined_contact': Combined contact labels (OR operation)
        - 'min_distances': Minimum distances for each joint
    """
    # Distance-based detection
    distance_contact = detect_contact_distance(
        hand_joints_3d, object_pose, object_mesh, contact_threshold
    )
    
    # Calculate actual distances for analysis
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(object_pose[3:7])
    rotation_matrix = rotation.as_matrix()
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = object_pose[:3]
    
    object_mesh_world = object_mesh.copy()
    object_mesh_world.apply_transform(transform_matrix)
    mesh_vertices = object_mesh_world.vertices
    
    distances = cdist(hand_joints_3d, mesh_vertices)
    min_distances = np.min(distances, axis=1)
    
    # Segmentation-based detection
    segmentation_contact = detect_contact_segmentation(
        hand_joints_2d, segmentation, object_seg_id
    )
    
    # Combined detection (OR operation)
    combined_contact = np.logical_or(distance_contact, segmentation_contact).astype(np.int32)
    
    return {
        'distance_contact': distance_contact,
        'segmentation_contact': segmentation_contact,
        'combined_contact': combined_contact,
        'min_distances': min_distances
    }


def get_mano_joint_names() -> List[str]:
    """Get MANO joint names."""
    return [
        "wrist",
        "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "little_mcp", "little_pip", "little_dip", "little_tip",
    ]


def analyze_contact_results(contact_results: Dict[str, np.ndarray]) -> None:
    """Analyze and print contact detection results."""
    joint_names = get_mano_joint_names()
    
    print("Contact Detection Results:")
    print("=" * 50)
    
    for method, contacts in contact_results.items():
        if method == 'min_distances':
            continue
            
        print(f"\n{method.upper()}:")
        contact_joints = np.where(contacts == 1)[0]
        if len(contact_joints) > 0:
            print(f"  Contact joints: {[joint_names[i] for i in contact_joints]}")
            print(f"  Total contacts: {len(contact_joints)}/21")
        else:
            print("  No contacts detected")
    
    if 'min_distances' in contact_results:
        print(f"\nMinimum distances:")
        for i, (name, dist) in enumerate(zip(joint_names, contact_results['min_distances'])):
            print(f"  {name}: {dist:.4f}m")


if __name__ == "__main__":
    # Example usage
    print("Contact detection utilities loaded successfully!")
    print("Available functions:")
    print("- detect_contact_distance()")
    print("- detect_contact_segmentation()")
    print("- detect_contact_combined()")
    print("- analyze_contact_results()")
