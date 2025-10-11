"""
SAPIEN-based contact detection for DexYCB dataset.
Uses mesh-based geometric contact detection with trimesh for accuracy.
"""

import numpy as np
import sapien
import trimesh
from typing import Dict, List, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation


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


def load_object_mesh(object_id: int, models_dir: Path) -> Optional[trimesh.Trimesh]:
    """
    Load YCB object mesh from textured_simple.obj file.
    
    Args:
        object_id: YCB object ID (1-21)
        models_dir: Path to DexYCB models directory
    
    Returns:
        Trimesh object or None if loading fails
    """
    try:
        if object_id not in YCB_CLASSES:
            print(f"Unknown object ID: {object_id}")
            return None
        
        object_name = YCB_CLASSES[object_id]
        mesh_file = models_dir / object_name / "textured_simple.obj"
        
        if not mesh_file.exists():
            print(f"Mesh file not found: {mesh_file}")
            return None
        
        # Load mesh using trimesh
        mesh = trimesh.load(str(mesh_file), force='mesh')
        
        # Ensure it's a Trimesh object (not a Scene)
        if isinstance(mesh, trimesh.Scene):
            # Get the first geometry from the scene
            mesh = list(mesh.geometry.values())[0]
        
        print(f"Loaded mesh for {object_name}:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Bounds: {mesh.bounds}")
        print(f"  Centroid: {mesh.centroid}")
        print(f"  Bounding sphere radius: {mesh.bounding_sphere.primitive.radius:.4f}m")
        
        return mesh
        
    except Exception as e:
        print(f"Error loading mesh for object {object_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_sphere_at_joint(
    scene: sapien.Scene,
    position: np.ndarray,
    radius: float = 0.005,
    name: str = "joint_sphere"
):
    """
    Create a sphere entity at joint position for contact detection.
    
    Args:
        scene: SAPIEN scene
        position: 3D position of the sphere
        radius: Radius of the sphere
        name: Name of the entity
    
    Returns:
        SAPIEN entity representing the sphere
    """
    builder = scene.create_actor_builder()
    builder.add_sphere_collision(sapien.Pose(), radius=radius)
    builder.add_sphere_visual(sapien.Pose(), radius=radius, color=[1.0, 0.0, 0.0, 0.5])
    sphere = builder.build_kinematic(name=name)
    sphere.set_pose(sapien.Pose(p=position))
    return sphere


def detect_contact_in_viewer(
    viewer,
    hand_joints_3d: np.ndarray,
    object_pose_7d: np.ndarray,
    object_id: int,
    models_dir: Path,
    object_idx: int = 0,
    sphere_radius: float = 0.005,
    distance_threshold: float = 0.02
) -> Dict[str, np.ndarray]:
    """
    Detect contact using mesh-based geometric method with real object mesh.
    
    Args:
        viewer: RobotHandDatasetSAPIENViewer instance (not used, kept for compatibility)
        hand_joints_3d: MANO hand joint positions in camera coordinates (21, 3)
        object_pose_7d: Object pose [x, y, z, qx, qy, qz, qw] from DexYCB data (camera frame)
        object_id: YCB object ID (1-21)
        models_dir: Path to DexYCB models directory
        object_idx: Index of the target object (not used, kept for compatibility)
        sphere_radius: Radius of sphere representing each joint (not used)
        distance_threshold: Distance threshold for contact (meters)
    
    Returns:
        Dictionary containing:
        - 'contact_labels': Binary array (21,) indicating contact for each joint
        - 'min_distances': Minimum distance from each joint to object surface
        - 'sphere_radius': Radius used
        - 'distance_threshold': Threshold used
        - 'mesh_radius': Actual bounding sphere radius of the mesh
    """
    joint_names = get_mano_joint_names()
    
    # Load object mesh
    print(f"\n{'='*60}")
    print(f"Loading mesh for object ID {object_id}...")
    mesh = load_object_mesh(object_id, models_dir)
    
    if mesh is None:
        print("ERROR: Failed to load object mesh!")
        return {
            'contact_labels': np.zeros(21, dtype=np.int32),
            'min_distances': np.full(21, -1.0, dtype=np.float32),
            'sphere_radius': sphere_radius,
            'distance_threshold': distance_threshold,
            'mesh_radius': -1.0,
            'error': 'Mesh loading failed'
        }
    
    # Transform mesh to camera frame using object pose
    object_position = object_pose_7d[:3]  # [x, y, z]
    object_quat = object_pose_7d[3:]      # [qx, qy, qz, qw]
    
    # Convert quaternion to rotation matrix
    rotation = Rotation.from_quat(object_quat).as_matrix()
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = object_position
    
    # Transform mesh to world frame
    mesh_transformed = mesh.copy()
    mesh_transformed.apply_transform(transform)
    
    print(f"\nMesh transformation:")
    print(f"  Original centroid: {mesh.centroid}")
    print(f"  Transformed centroid: {mesh_transformed.centroid}")
    print(f"  Object position from pose: {object_position}")
    print(f"  Bounding sphere radius: {mesh.bounding_sphere.primitive.radius:.4f}m")
    
    # Initialize arrays
    contact_labels = np.zeros(21, dtype=np.int32)
    min_distances = np.zeros(21, dtype=np.float32)
    
    print(f"\n{'='*60}")
    print(f"Computing distances from hand joints to mesh surface...")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"{'='*60}")
    
    num_contacts = 0
    for i, (joint_pos, joint_name) in enumerate(zip(hand_joints_3d, joint_names)):
        # Skip invalid joints
        if np.all(joint_pos == -1):
            min_distances[i] = -1
            print(f"  Joint {i:2d} ({joint_name:12s}): INVALID (position = -1)")
            continue
        
        # Calculate closest point on mesh surface and distance
        closest_point, distance, triangle_id = mesh_transformed.nearest.on_surface([joint_pos])
        distance = distance[0]  # Extract scalar from array
        
        min_distances[i] = distance
        
        # Check if in contact (distance <= threshold)
        if distance <= distance_threshold:
            contact_labels[i] = 1
            num_contacts += 1
            print(f"  Joint {i:2d} ({joint_name:12s}): distance = {distance:.4f}m <= {distance_threshold:.4f}m → ✓ CONTACT")
        else:
            print(f"  Joint {i:2d} ({joint_name:12s}): distance = {distance:.4f}m > {distance_threshold:.4f}m")
    
    print(f"\n{'='*60}")
    print(f"Total contacts detected: {num_contacts}/21")
    print(f"{'='*60}\n")
    
    return {
        'contact_labels': contact_labels,
        'min_distances': min_distances,
        'distances_to_center': min_distances,  # Kept for compatibility
        'sphere_radius': sphere_radius,
        'distance_threshold': distance_threshold,
        'mesh_radius': mesh.bounding_sphere.primitive.radius,
    }


def detect_contact_sapien_simple(
    viewer,
    hand_joints_3d: np.ndarray,
    object_idx: int = 0,
    contact_threshold: float = 0.02
) -> Dict[str, np.ndarray]:
    """
    Simplified contact detection using distance to object mesh.
    Does not create temporary actors, just checks distances.
    
    Args:
        viewer: RobotHandDatasetSAPIENViewer instance
        hand_joints_3d: MANO hand joint positions (21, 3)
        object_idx: Index of target object
        contact_threshold: Distance threshold for contact
    
    Returns:
        Dictionary with contact information
    """
    from scipy.spatial.distance import cdist
    
    # Get the target object
    if object_idx >= len(viewer.objects):
        raise ValueError(f"Object index {object_idx} out of range")
    
    target_object = viewer.objects[object_idx]
    
    # Try to get object mesh vertices
    # SAPIEN objects may not directly expose mesh, so we use bounding box as approximation
    contact_labels = np.zeros(21, dtype=np.int32)
    min_distances = np.zeros(21, dtype=np.float32)
    
    object_pose = target_object.get_pose()
    object_center = object_pose.p
    
    for i, joint_pos in enumerate(hand_joints_3d):
        # Skip invalid joints
        if np.all(joint_pos == -1):
            min_distances[i] = -1
            continue
        
        # Calculate distance to object center as approximation
        distance = np.linalg.norm(joint_pos - object_center)
        min_distances[i] = distance
        
        # Determine contact based on threshold
        if distance <= contact_threshold:
            contact_labels[i] = 1
    
    return {
        'contact_labels': contact_labels,
        'min_distances': min_distances,
        'contact_threshold': contact_threshold
    }


def get_contact_summary(contact_labels: np.ndarray) -> Dict:
    """
    Generate a summary of contact results.
    
    Args:
        contact_labels: Binary array (21,) of contact labels
    
    Returns:
        Dictionary with contact summary
    """
    joint_names = get_mano_joint_names()
    contact_indices = np.where(contact_labels == 1)[0]
    contact_joint_names = [joint_names[i] for i in contact_indices]
    
    # Group by finger
    fingers = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'little': [17, 18, 19, 20],
        'wrist': [0]
    }
    
    finger_contacts = {}
    for finger_name, joint_indices in fingers.items():
        finger_contact_count = sum(contact_labels[i] for i in joint_indices)
        finger_contacts[f"{finger_name}_contacts"] = finger_contact_count
    
    return {
        'total_contacts': len(contact_indices),
        'contact_joint_indices': contact_indices.tolist(),
        'contact_joint_names': contact_joint_names,
        **finger_contacts
    }


def analyze_contact_results(contact_results: Dict) -> None:
    """
    Analyze and print SAPIEN contact detection results.
    
    Args:
        contact_results: Results from detect_contact_in_viewer or similar
    """
    joint_names = get_mano_joint_names()
    contact_labels = contact_results['contact_labels']
    
    print("SAPIEN Contact Detection Results:")
    print("=" * 60)
    
    contact_indices = np.where(contact_labels == 1)[0]
    if len(contact_indices) > 0:
        print(f"Contact joints ({len(contact_indices)}/21):")
        for idx in contact_indices:
            joint_name = joint_names[idx]
            if 'distances_to_center' in contact_results:
                dist = contact_results['distances_to_center'][idx]
                print(f"  [{idx:2d}] {joint_name:15s} - distance: {dist:.4f}m")
            else:
                print(f"  [{idx:2d}] {joint_name:15s}")
    else:
        print("No contacts detected")
    
    # Print summary by finger
    summary = get_contact_summary(contact_labels)
    print("\nContacts by finger:")
    for finger in ['wrist', 'thumb', 'index', 'middle', 'ring', 'little']:
        key = f"{finger}_contacts"
        if key in summary:
            print(f"  {finger:10s}: {summary[key]}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("SAPIEN contact detection utilities loaded successfully!")
    print("Available functions:")
    print("- detect_contact_in_viewer()")
    print("- detect_contact_sapien_simple()")
    print("- get_contact_summary()")
    print("- analyze_contact_results()")

