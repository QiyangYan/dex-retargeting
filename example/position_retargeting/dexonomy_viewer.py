"""
SAPIEN viewer for Dexonomy grasp dataset with Shadow Hand and OmniHand
"""
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import sapien
from tqdm import trange
from termcolor import cprint
from scipy.spatial.transform import Rotation

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.robot_wrapper import RobotWrapper


class DexonomyGraspSAPIENViewer:
    """Viewer for Dexonomy grasp dataset with Shadow Hand and retargeting to other hands"""
    
    def __init__(
        self,
        robot_names: List[RobotName],
        headless: bool = False,
        use_ray_tracing: bool = False,
        show_ground: bool = False,
        show_table: bool = False,
        data_root: Optional[Path] = None,
        hand_type: HandType = HandType.right,
        retargeting_type: RetargetingType = RetargetingType.vector,
        two_optimizers: bool = False,
        second_optimizer_type: str = "FINGERTIP",
    ):
        """
        Args:
            robot_names: List of robot hand names to visualize (e.g., [RobotName.shadow, RobotName.allegro])
            headless: Whether to run in headless mode (for video recording)
            use_ray_tracing: Whether to use ray tracing for rendering
            show_ground: Whether to show ground plane (default: False)
            show_table: Whether to show table (default: False)
            data_root: Root directory of Dexonomy dataset (for loading camera poses)
            hand_type: HandType (default: HandType.right)
            retargeting_type: RetargetingType for first optimizer (default: RetargetingType.vector)
            two_optimizers: Whether to use two optimizers (default: False)
            second_optimizer_type: Type of second optimizer ("VECTOR" or "FINGERTIP")
        """
        self.headless = headless
        self.robot_names = robot_names
        self.show_ground = show_ground
        self.show_table = show_table
        self.hand_type = hand_type
        self.two_optimizers = two_optimizers
        self.second_optimizer_type = second_optimizer_type
        
        cprint(f"[INFO] Hand type: {hand_type}", "green")
        cprint(f"[INFO] Retargeting type: {retargeting_type}", "green")
        cprint(f"[INFO] Two optimizers: {two_optimizers}", "green")
        if two_optimizers:
            cprint(f"[INFO] Second optimizer type: {second_optimizer_type}", "green")
        
        # Store data root for camera pose loading
        if data_root is None:
            data_root = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
        self.data_root = data_root
        
        # Create SAPIEN engine and scene
        if use_ray_tracing:
            self.engine = sapien.Engine()
            sapien.render.set_ray_tracing_denoiser("optix")
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(8)
        else:
            self.engine = sapien.Engine()
        
        render_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(render_config)
        self.scene.set_timestep(1 / 240)
        
        # Add lighting
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, -1, -1], [0.5, 0.5, 0.5], True)
        self.scene.add_directional_light([1, 1, -1], [0.5, 0.5, 0.5], False)
        
        # Add ground if requested
        if self.show_ground:
            self.scene.add_ground(altitude=0, render=True)
        
        # Add table if requested
        if self.show_table:
            self._add_table()
        else:
            self.table = None  # No table
        
        # Setup camera/viewer
        if not headless:
            self.viewer = self.scene.create_viewer()
            self.viewer.set_camera_xyz(x=0.8, y=0, z=0.6)
            self.viewer.set_camera_rpy(r=0, p=-0.3, y=0)
            self.viewer.paused = False
        else:
            # Setup camera for headless rendering
            self.camera = self.scene.add_camera(
                name="viewer_camera",
                width=1920,
                height=1080,
                fovy=np.deg2rad(45),
                near=0.1,
                far=100,
            )
            self.camera.set_local_pose(sapien.Pose([0.8, 0, 0.6], [0.9239, 0, -0.3827, 0]))
        
        # Load robots
        self.robots: List[sapien.Articulation] = []
        self.retargetings: List[Optional[SeqRetargeting]] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.second_retargeting: Optional[SeqRetargeting] = None
        self._load_robots(retargeting_type)
        
        # Shadow Hand FK wrapper for computing joint positions
        self.shadow_robot_wrapper: Optional[RobotWrapper] = None
        if len(self.robots) > 0:
            # Load Shadow Hand URDF for FK computation
            shadow_config_path = get_default_config_path(
                RobotName.shadow_no_wrist, RetargetingType.position, hand_type
            )
            shadow_config = RetargetingConfig.load_from_file(shadow_config_path)
            self.shadow_robot_wrapper = RobotWrapper(shadow_config.urdf_path)
            cprint(f"[INFO] Loaded Shadow Hand FK wrapper for joint position extraction", "cyan")
        
        # Objects will be loaded dynamically
        self.objects: List[sapien.Actor] = []
        
        # Joint position markers (spheres)
        self.joint_markers: List[sapien.Actor] = []
    
    def _add_table(self):
        """Add a table to the scene"""
        # Create table material
        table_material = sapien.render.RenderMaterial()
        table_material.set_base_color(np.array([0.8, 0.6, 0.4, 1]))
        table_material.set_roughness(0.8)
        table_material.set_metallic(0.0)
        table_material.set_specular(0.5)
        
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.4, 0.4, 0.025])
        builder.add_box_visual(half_size=[0.4, 0.4, 0.025], material=table_material)
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, -0.025]))
    
    def _create_joint_marker_spheres(self, joint_positions: np.ndarray, color: np.ndarray = None):
        """
        Create sphere markers at joint positions
        
        Args:
            joint_positions: (N, 3) array of joint positions in world frame
            color: (4,) RGBA color array (default: red [1, 0, 0, 0.8])
        """
        # Clear existing markers
        for marker in self.joint_markers:
            self.scene.remove_actor(marker)
        self.joint_markers.clear()
        
        # Default color: semi-transparent red
        if color is None:
            color = np.array([1.0, 0.0, 0.0, 0.8])
        
        # Create sphere material
        sphere_material = sapien.render.RenderMaterial()
        sphere_material.set_base_color(color)
        sphere_material.set_roughness(0.3)
        sphere_material.set_metallic(0.1)
        sphere_material.set_specular(0.8)
        
        # Sphere radius: 1cm = 0.01m
        sphere_radius = 0.01
        
        # Create a sphere at each joint position
        for i, pos in enumerate(joint_positions):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=sphere_radius, material=sphere_material)
            marker = builder.build_static(name=f"joint_marker_{i}")
            marker.set_pose(sapien.Pose(pos))
            self.joint_markers.append(marker)
        
        cprint(f"[INFO] Created {len(joint_positions)} joint markers (spheres, radius={sphere_radius}m)", "cyan")
    
    def _load_robots(self, retargeting_type: RetargetingType):
        """Load all robot hands"""
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        
        for i, robot_name in enumerate(self.robot_names):
            cprint(f"[INFO] Loading robot: {robot_name}", "green")
            
            # First robot is Shadow Hand, use qpos directly (no retargeting)
            if i == 0:
                # 直接加载 Shadow Hand URDF（不需要通过 config，它是 retargeting 的参考源）
                shadow_urdf_dir = Path(__file__).parent.parent.parent / "assets" / "robots" / "hands" / "shadow_hand_no_wrist"
                hand_suffix = "right"
                urdf_path = shadow_urdf_dir / f"shadow_hand_{hand_suffix}.urdf"
                
                cprint(f"[INFO] Loading Shadow Hand URDF: {urdf_path}", "white")
                
                # 使用 yourdfpy 加载并添加 dummy joints
                robot_urdf = urdf.URDF.load(
                    str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
                )
                
                # 写入临时文件
                urdf_name = urdf_path.name
                temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
                temp_path = f"{temp_dir}/{urdf_name}"
                robot_urdf.write_xml_file(temp_path)
                
                # 使用 SAPIEN loader 加载
                robot = loader.load(temp_path)
                self.robots.append(robot)
                self.retargetings.append(None)  # Shadow Hand 不需要 retargeting
                
                # 获取关节映射
                sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
                cprint(f"[INFO] Shadow Hand joint names: {sapien_joint_names}", "white")
                self.retarget2sapien.append(np.arange(len(sapien_joint_names)))
                
            else:
                # Other robots need retargeting from Shadow Hand
                cprint(f"[INFO] Setting up retargeting for {robot_name}", "cyan")
                
                # Setup second optimizer if enabled
                if self.two_optimizers and self.second_retargeting is None:
                    if self.second_optimizer_type == "VECTOR":
                        second_retargeting_type = RetargetingType.vector
                    elif self.second_optimizer_type == "FINGERTIP":
                        second_retargeting_type = RetargetingType.fingertip
                    elif self.second_optimizer_type == "DEXPILOT":
                        second_retargeting_type = RetargetingType.dexpilot
                    elif self.second_optimizer_type == "POSITION":
                        second_retargeting_type = RetargetingType.position
                    else:
                        raise ValueError(f"Unsupported second optimizer type: {self.second_optimizer_type}")
                    
                    second_config_path = get_default_config_path(
                        robot_name, second_retargeting_type, self.hand_type
                    )
                    override = dict(add_dummy_free_joint=True)
                    second_config = RetargetingConfig.load_from_file(second_config_path, override=override)
                    self.second_retargeting = second_config.build()
                    cprint(f"[INFO] Second optimizer ({self.second_optimizer_type}) loaded", "green")
                
                # Setup first optimizer
                config_path = get_default_config_path(
                    robot_name, retargeting_type, self.hand_type
                )
                override = dict(add_dummy_free_joint=True)
                config = RetargetingConfig.load_from_file(config_path, override=override)
                retargeting = config.build()
                self.retargetings.append(retargeting)
                
                # Build robot with glb suffix
                urdf_path = Path(config.urdf_path)
                if "glb" not in urdf_path.stem:
                    urdf_path = urdf_path.with_name(urdf_path.stem + "_glb" + urdf_path.suffix)
                
                cprint(f"[INFO] Loading URDF: {urdf_path}", "white")
                robot_urdf = urdf.URDF.load(
                    str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
                )
                urdf_name = urdf_path.name
                temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
                temp_path = f"{temp_dir}/{urdf_name}"
                robot_urdf.write_xml_file(temp_path)
                
                robot = loader.load(temp_path)
                self.robots.append(robot)
                
                sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
                cprint(f"[INFO] {robot_name} joint count: {len(sapien_joint_names)}", "white")
                retarget2sapien = np.array(
                    [retargeting.joint_names.index(n) for n in sapien_joint_names]
                ).astype(int)
                self.retarget2sapien.append(retarget2sapien)
            
            cprint(f"[INFO] Robot {i} loaded with {self.robots[i].dof} DOF", "cyan")
    
    def _compute_shadow_joint_positions(self, shadow_qpos: np.ndarray) -> np.ndarray:
        """
        Compute Shadow Hand joint positions from qpos using FK
        
        Args:
            shadow_qpos: (29,) Shadow Hand qpos
            
        Returns:
            joint_positions: (N, 3) array of joint positions in world frame
        """
        if self.shadow_robot_wrapper is None:
            raise ValueError("Shadow Hand FK wrapper not initialized")
        
        # Extract joint angles (ignore base pose for FK computation)
        joint_angles = shadow_qpos[7:29]  # 22 joint angles
        
        # Create full qpos for FK (without dummy free joints)
        qpos_for_fk = joint_angles
        
        # Compute FK
        self.shadow_robot_wrapper.compute_forward_kinematics(qpos_for_fk)
        
        # Get link positions for retargeting
        # We need to extract positions to match MANO hand's 21 joints
        # MANO joint order: wrist, thumb(mcp,pip,dip,tip), index(mcp,pip,dip,tip), 
        #                   middle(mcp,pip,dip,tip), ring(mcp,pip,dip,tip), little(mcp,pip,dip,tip)
        # Shadow Hand URDF link naming convention (without "rh_" prefix)
        link_names = [
            # 0: wrist
            "palm",
            # 1-4: thumb (mcp, pip, dip, tip)
            "thbase", "thproximal", "thmiddle", "thtip",
            # 5-8: index (mcp, pip, dip, tip)
            "ffknuckle", "ffproximal", "ffmiddle", "fftip",
            # 9-12: middle (mcp, pip, dip, tip)
            "mfknuckle", "mfproximal", "mfmiddle", "mftip",
            # 13-16: ring (mcp, pip, dip, tip)
            "rfknuckle", "rfproximal", "rfmiddle", "rftip",
            # 17-20: little/pinky (mcp, pip, dip, tip)
            "lfknuckle", "lfproximal", "lfmiddle", "lftip",
        ]
        
        joint_positions = []
        for link_name in link_names:
            try:
                link_id = self.shadow_robot_wrapper.get_link_index(link_name)
                link_pose = self.shadow_robot_wrapper.get_link_pose(link_id)
                position = link_pose[:3, 3]
                joint_positions.append(position)
            except ValueError:
                cprint(f"[WARNING] Link {link_name} not found in Shadow Hand URDF", "yellow")
                joint_positions.append(np.zeros(3))
        
        joint_positions = np.array(joint_positions)
        
        # Transform to world frame using base pose
        base_pos = shadow_qpos[0:3]
        base_quat_wxyz = shadow_qpos[3:7]
        
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], 
                               base_quat_wxyz[3], base_quat_wxyz[0]])
        rot_matrix = r.as_matrix()
        
        # Transform all positions to world frame
        joint_positions_world = (rot_matrix @ joint_positions.T).T + base_pos
        
        return joint_positions_world
    
    def _compute_shadow_joint_positions_world(self, shadow_qpos: np.ndarray) -> np.ndarray:
        """
        Compute Shadow Hand joint positions in world frame
        
        Args:
            shadow_qpos: (29,) Shadow Hand qpos
            
        Returns:
            joint_positions_world: (N, 3) array of joint positions in world frame
        """
        # Get joint positions in local frame
        joint_positions_local = self._compute_shadow_joint_positions(shadow_qpos)
        
        # Extract base pose
        base_pos = shadow_qpos[0:3]
        base_quat_wxyz = shadow_qpos[3:7]
        
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], 
                               base_quat_wxyz[3], base_quat_wxyz[0]])
        rot_matrix = r.as_matrix()
        
        # Transform all positions to world frame
        joint_positions_world = (rot_matrix @ joint_positions_local.T).T + base_pos
        
        return joint_positions_world
    
    def _get_camera_pose(self, data: Dict) -> Optional[sapien.Pose]:
        """
        Try to load camera pose from dataset
        
        Args:
            data: Dataset sample dictionary
            
        Returns:
            Camera pose if available, None otherwise
        """
        try:
            object_id = data["object_id"]
            scale_name = data["scale_name"]
            
            # Path to vision data
            vision_dir = self.data_root / "objaverse_5k" / "vision_data" / "azure_kinect_dk" / object_id / "floating" / scale_name
            
            if not vision_dir.exists():
                return None
            
            # Load first camera extrinsic (cam_ex_00.npy)
            cam_ex_file = vision_dir / "cam_ex_00.npy"
            if not cam_ex_file.exists():
                return None
            
            # Load camera extrinsic matrix (4x4)
            cam_ex = np.load(cam_ex_file, allow_pickle=True)
            
            # Convert to SAPIEN pose
            # Note: This camera extrinsic has position in the last row (not column)
            position = cam_ex[3, :3]  # Position is in the last row, first 3 elements
            rotation_matrix = cam_ex[:3, :3]
            
            # Convert rotation matrix to quaternion (wxyz format for SAPIEN)
            r = Rotation.from_matrix(rotation_matrix)
            quat_xyzw = r.as_quat()  # scipy gives xyzw
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            camera_pose = sapien.Pose(position, quat_wxyz)
            
            cprint(f"[INFO] Loaded camera pose from dataset", "cyan")
            cprint(f"  Position: {position}", "white")
            
            return camera_pose
            
        except Exception as e:
            cprint(f"[WARNING] Failed to load camera pose: {e}", "yellow")
            return None
    
    def load_object(self, data: Dict):
        """
        Load object from Dexonomy data with correct scale
        Load one object copy for each robot (Shadow + retargeted robots)
        
        Args:
            data: Dataset sample dictionary (contains single grasp)
        """
        # Clear previous objects
        for obj in self.objects:
            self.scene.remove_actor(obj)
        self.objects = []
        
        # Load object mesh
        mesh_path = data["object_mesh_path"]
        object_id = data["object_id"]
        
        # Get object configuration for scale and pose
        scene_config = data["scene_config"]
        obj_config = scene_config[object_id]
        base_scale = obj_config["scale"]  # Base scale from config (e.g., [0.05, 0.05, 0.05])
        obj_pose = obj_config["pose"]
        
        # Extract position and quaternion from config
        position = np.array(obj_pose[:3])  # [x, y, z]
        quat_wxyz = np.array(obj_pose[3:])  # [qw, qx, qy, qz] - wxyz format
        
        # Use base scale with scene_scale
        actual_scale = base_scale
        
        # Load one object copy for each robot
        num_robots = len(self.robots)
        for robot_idx in range(num_robots):
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(
                filename=mesh_path,
                scale=actual_scale * data["scene_scale"],
            )
            
            obj = builder.build_static(name=f"{object_id}_{robot_idx}")
            self.objects.append(obj)
            
            # Set object pose (offset will be applied in render_grasp_single)
            obj_pose_sapien = sapien.Pose(position, quat_wxyz)
            obj.set_pose(obj_pose_sapien)
        
        cprint(f"[INFO] Loaded {num_robots} copies of object: {object_id}", "yellow")
        cprint(f"  Object scale: {actual_scale}", "white")
        cprint(f"  Base pose: position={position}, quat_wxyz={quat_wxyz}", "white")
        cprint(f"  Scene scale (applied to object): {data['scene_scale']}", "white")
    
    def set_shadow_qpos(self, shadow_qpos: np.ndarray, robot_idx: int = 0, y_offset: float = 0.0):
        """
        Set Shadow Hand qpos from Dexonomy dataset using dummy joints
        
        Dexonomy 29D format:
        - [0:3]: base translation (x, y, z)
        - [3:7]: base quaternion (w, x, y, z)
        - [7:29]: 22 joint angles (Shadow Hand without wrist)
        
        Args:
            shadow_qpos: (29,) array from Dexonomy dataset
            robot_idx: Which robot to set (should be 0 for Shadow Hand)
            y_offset: Y-axis offset for parallel display (default: 0.0)
        """
        if robot_idx >= len(self.robots):
            return
        
        robot = self.robots[robot_idx]
        expected_dof = robot.dof
        
        # Parse 29D data
        base_pos = shadow_qpos[0:3].copy()  # xyz
        base_pos[1] += y_offset  # Apply y offset
        base_quat_wxyz = shadow_qpos[3:7]  # quaternion as (w, x, y, z)
        joint_angles = shadow_qpos[7:29]  # 22 joint angles
      
        # Convert quaternion to euler angles (for dummy rotation joints)
        # SAPIEN expects wxyz format, scipy uses xyzw
        r = Rotation.from_quat([base_quat_wxyz[1], base_quat_wxyz[2], 
                               base_quat_wxyz[3], base_quat_wxyz[0]])  # scipy uses xyzw
        
        base_euler = r.as_euler('XYZ', degrees=False)  # Get x,y,z rotations in radians

        # Set joint angles through qpos (including base pose via dummy joints)
        if expected_dof == 28:
            # Robot has 28 DOF:
            # - [0-2]: dummy_x/y/z_translation_joint (base position)
            # - [3-5]: dummy_x/y/z_rotation_joint (base rotation as euler angles)
            # - [6-27]: 22 other hand joints
            full_qpos = np.zeros(28, dtype=np.float32)
            # Dummy translation joints (indices 0-2): set base position with offset
            full_qpos[0:3] = base_pos
            # Dummy rotation joints (indices 3-5): set base rotation (euler angles)
            full_qpos[3:6] = base_euler
            # Hand joints (indices 6-27): set from dataset (22 joint angles)
            full_qpos[6:28] = joint_angles
            
            robot.set_qpos(full_qpos)
        elif expected_dof == 22:
            # No dummy joints, only hand joints
            full_qpos = np.zeros(22, dtype=np.float32)
            full_qpos[:] = joint_angles
            robot.set_qpos(full_qpos)
            cprint(f"[WARNING] Robot has no dummy joints, cannot apply offset", "yellow")
        else:
            cprint(f"[WARNING] Unexpected robot DOF: {expected_dof}, expected 28", "yellow")
            # Fallback: try to set what we have
            if expected_dof >= len(joint_angles):
                qpos = np.zeros(expected_dof, dtype=np.float32)
                qpos[:len(joint_angles)] = joint_angles
                robot.set_qpos(qpos)
    
    def retarget_to_robot(self, shadow_qpos: np.ndarray, robot_idx: int, y_offset: float = 0.0):
        """
        Retarget Shadow Hand pose to another robot using FK and optimization
        
        Args:
            shadow_qpos: (29,) Shadow Hand qpos
            robot_idx: Which robot to retarget to (should be >= 1)
            y_offset: Y-axis offset for parallel display (default: 0.0)
        """
        if robot_idx == 0 or robot_idx >= len(self.robots):
            return
        
        retargeting = self.retargetings[robot_idx]
        if retargeting is None:
            return
        
        # Compute Shadow Hand joint positions using FK
        joint_positions = self._compute_shadow_joint_positions(shadow_qpos)
        
        # Get reference value based on retargeting type (first optimizer)
        retargeting_type = retargeting.optimizer.retargeting_type
        indices = retargeting.optimizer.target_link_human_indices
        
        cprint(f"[DEBUG] Retargeting type: {retargeting_type}", "cyan")
        cprint(f"[DEBUG] Joint positions shape: {joint_positions.shape}", "white")
        cprint(f"[DEBUG] Indices shape: {indices.shape}", "white")
        
        if retargeting_type == "POSITION":
            ref_value = joint_positions[indices, :]
        elif retargeting_type == "FINGERTIP":
            ref_value = joint_positions[indices, :]
        elif retargeting_type == "DEXPILOT":
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_positions[task_indices, :] - joint_positions[origin_indices, :]
        elif retargeting_type == "VECTOR":
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_positions[task_indices, :] - joint_positions[origin_indices, :]
        else:
            ref_value = joint_positions[indices, :]
        
        # Align omni hand with shadow hand palm frame
        canonical_frame = retargeting.optimizer.canonical_frame
        shadow_rotation = Rotation.from_quat([shadow_qpos[4], shadow_qpos[5], shadow_qpos[6], shadow_qpos[3],])
        robot_r = shadow_rotation * Rotation.from_matrix(canonical_frame)

        # Get full retargeting output (first optimizer)
        last_pos = np.concatenate([shadow_qpos[:3], robot_r.as_euler("XYZ", degrees=False), retargeting.mean_qpos[6:]])
        qpos_full = retargeting.retarget(ref_value, last_qpos=last_pos)
        qpos = qpos_full[self.retarget2sapien[robot_idx]]
        
        # Second optimizer for finger refinement
        if self.two_optimizers and self.second_retargeting is not None:
            second_retargeting_type = self.second_retargeting.optimizer.retargeting_type
            second_indices = self.second_retargeting.optimizer.target_link_human_indices
            
            cprint(f"[DEBUG] Second optimizer type: {second_retargeting_type}", "cyan")
            
            if second_retargeting_type == "POSITION":
                second_ref_value = joint_positions[second_indices, :]
            elif second_retargeting_type == "FINGERTIP":
                second_ref_value = joint_positions[second_indices, :]
            elif second_retargeting_type == "DEXPILOT":
                second_origin_indices = second_indices[0, :]
                second_task_indices = second_indices[1, :]
                second_ref_value = joint_positions[second_task_indices, :] - joint_positions[second_origin_indices, :]
            else:  # VECTOR
                second_origin_indices = second_indices[0, :]
                second_task_indices = second_indices[1, :]
                second_ref_value = joint_positions[second_task_indices, :] - joint_positions[second_origin_indices, :]
            
            # Update second optimizer's last_qpos with first optimizer's result
            # Extract only the target joints (idx_pin2target) from the full qpos
            self.second_retargeting.last_qpos = qpos_full[self.second_retargeting.optimizer.idx_pin2target]
            
            # Retarget with second optimizer
            qpos_second = self.second_retargeting.retarget(second_ref_value)[self.retarget2sapien[robot_idx]]
            qpos = qpos_second  # Use second optimizer result
            # import ipdb; ipdb.set_trace()
            cprint(f"[DEBUG] Applied second optimizer result", "green")
        
        # Apply y offset to the retargeted robot
        # If robot has dummy joints (indices 0-5), modify them
        robot = self.robots[robot_idx]
        if robot.dof >= 6:
            # Assume first 3 DOF are translation, next 3 are rotation
            qpos[1] += y_offset  # Apply y offset to translation
        
        # Set joint angles
        robot.set_qpos(qpos.astype(np.float32))
    
    def render_grasp_single(self, data: Dict, fps: int = 10, y_offset: float = 0.0):
        """
        Render a single grasp from Dexonomy data with parallel display
        
        Args:
            data: Dictionary from DexonomyGraspDataset (single grasp)
            fps: Frames per second
            y_offset: Y-axis spacing between robots (default: 0.5m)
        """
        # Get grasp data (only grasp phase)
        grasp_qpos = data["grasp_qpos"]  # (29,)
        grasp_qpos_pin_order = data["grasp_qpos_pin_order"]  # (29,)
        scene_config = data["scene_config"]
        object_id = data["object_id"]
        num_robots = len(self.robots)
        
        # Calculate camera position to center all robots
        # Shadow Hand at y=0, other robots at y=offset*idx
        # Center = (0 + y_offset * (num_robots-1)) / 2
        if num_robots > 1:
            global_y_offset = y_offset * (num_robots - 1) / 2.0
        else:
            global_y_offset = 0.0
        
        # Adjust camera to see all robots
        if not self.headless:
            try:
                self.viewer.set_camera_xyz(0.4, global_y_offset, 0.3)
                self.viewer.set_camera_rpy(0, -0.4, 0)
            except AttributeError:
                # Viewer not fully initialized, skip camera setup
                pass
        else:
            self.camera.set_local_pose(
                sapien.Pose([0.4, global_y_offset, 0.3], [0.9808, 0, -0.1951, 0])
            )
        
        # Position table if it exists (center it)
        if self.table is not None:
            self.table.set_pose(sapien.Pose([0.0, global_y_offset, -0.15]))
        
        # Setup video writer if headless
        if self.headless:
            robot_names_str = "_".join([str(r) for r in self.robot_names])
            video_path = Path(__file__).parent / f"dexonomy_{robot_names_str}_{data['grasp_type']}.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (self.camera.get_width(), self.camera.get_height()),
            )
            cprint(f"[INFO] Saving video to {video_path}", "cyan")
        
        # Load objects (one copy per robot)
        self.load_object(data)
        
        # Display info
        cprint(f"\n[INFO] Rendering grasp:", "green")
        cprint(f"  Grasp Type: {data['grasp_type']}", "yellow")
        cprint(f"  Object ID: {object_id}", "yellow")
        cprint(f"  Scale Name: {data['scale_name']}", "yellow")
        cprint(f"  Grasp Index: {data['grasp_idx']}/{data['num_grasps_in_file']}", "yellow")
        cprint(f"  Number of robots: {num_robots}", "yellow")
        cprint(f"  Y-axis spacing: {y_offset}m", "yellow")
        
        # Render only the grasp phase (static pose)
        cprint(f"\n[INFO] Starting visualization (press ESC to quit)...", "cyan")
        cprint(f"  Showing grasp pose with parallel display", "white")
        
        # Shadow Hand (robot_idx=0) stays at original pose (no offset)
        # Other robots get offsets for parallel display
        cprint(f"  Robot 0 (Shadow Hand): y_offset = 0.00m (original dataset pose)", "white")
        self.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
        
        # Shadow Hand object stays at original pose (no offset)
        # Already set in load_object, no need to modify
        
        # Retarget to other robots with offsets
        for robot_idx in range(1, num_robots):
            # Calculate offset for retargeted robots (starts from y_offset)
            offset_y = y_offset * robot_idx
            cprint(f"  Robot {robot_idx} ({self.robot_names[robot_idx]}): y_offset = {offset_y:.2f}m", "white")
            
            cprint(f"[INFO] Retargeting to robot {robot_idx} ({self.robot_names[robot_idx]})", "cyan")
            self.retarget_to_robot(grasp_qpos_pin_order, robot_idx=robot_idx, y_offset=offset_y)
            
            # Update object pose for this retargeted robot
            if robot_idx < len(self.objects):
                # Get base object pose and apply offset
                obj_pose = self.objects[robot_idx].get_pose()
                pose_offset = sapien.Pose([0, offset_y, 0])
                new_pose = pose_offset * obj_pose
                self.objects[robot_idx].set_pose(new_pose)
        
        # Visualize joint positions as spheres (1cm radius)
        cprint(f"[INFO] Creating joint position markers...", "cyan")
        joint_positions_world = self._compute_shadow_joint_positions(grasp_qpos_pin_order)
        cprint(f"[DEBUG] Joint positions world shape: {joint_positions_world.shape}", "white")
        cprint(f"[DEBUG] Sample joint position: {joint_positions_world[0]}", "white")
        self._create_joint_marker_spheres(joint_positions_world)
        
        self.scene.update_render()
        
        # Keep rendering until user closes or in headless mode
        if self.headless:
            # In headless mode, render a few frames for video
            frames_to_render = 30  # Render 30 frames (3 seconds at 10 fps)
            for frame in range(frames_to_render):
                self.scene.update_render()
                self.camera.take_picture()
                rgb = self.camera.get_picture("Color")[..., :3]
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                writer.write(rgb[..., ::-1])
        else:
            # In interactive mode, keep rendering until viewer is closed
            while not self.viewer.closed:
                self.viewer.render()
        
        if self.headless:
            writer.release()
            cprint(f"\n[INFO] Video saved successfully to {video_path}!", "green")
        else:
            cprint(f"\n[INFO] Visualization complete! (press ESC to close)", "green")

