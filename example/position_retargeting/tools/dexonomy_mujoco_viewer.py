"""
MuJoCo viewer for Dexonomy grasp dataset with Shadow Hand
"""
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import mujoco
import mujoco.viewer
from tqdm import trange
from termcolor import cprint
from scipy.spatial.transform import Rotation


class DexonomyGraspMuJoCoViewer:
    """Viewer for Dexonomy grasp dataset with Shadow Hand using MuJoCo"""
    
    def __init__(
        self,
        headless: bool = False,
        show_ground: bool = False,
        show_table: bool = False,
        data_root: Optional[Path] = None,
        use_tabletop_pose: bool = True,
        xml_path: Optional[str] = None,
    ):
        """
        Args:
            headless: Whether to run in headless mode (for video recording)
            show_ground: Whether to show ground plane (default: False)
            show_table: Whether to show table (default: False)
            data_root: Root directory of Dexonomy dataset
            use_tabletop_pose: Whether to use tabletop_pose.json (True) or origin pose (False)
            xml_path: Path to Shadow Hand XML file
        """
        self.headless = headless
        self.show_ground = show_ground
        self.show_table = show_table
        self.use_tabletop_pose = use_tabletop_pose
        
        # Store data root for camera pose loading
        if data_root is None:
            data_root = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
        self.data_root = data_root
        
        # Default XML path
        if xml_path is None:
            xml_path = "/home/guizhewei/guizhewei/retarget/dex-retargeting/assets/robots/hands/shadow_hand_noforearm_xml/right_hand.xml"
        self.xml_path = xml_path
        
        # MuJoCo model and data will be created dynamically
        self.model = None
        self.data = None
        self.viewer = None
        
        # Store current object info
        self.current_object_mesh = None
        self.current_object_scale = None
        self.current_object_pose = None
        
    def _create_base_xml(self) -> ET.Element:
        """Create base MuJoCo XML with hand, optional ground, and table"""
        # Load Shadow Hand XML
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        # Fix mesh directory path to absolute path
        # The XML has meshdir="assets" which is relative to the XML file
        compiler = root.find('compiler')
        if compiler is not None:
            xml_dir = Path(self.xml_path).parent
            meshdir = compiler.get('meshdir', 'assets')
            # Convert to absolute path
            abs_meshdir = xml_dir / meshdir
            compiler.set('meshdir', str(abs_meshdir))
        
        # Find or create worldbody
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')
        
        # Add freejoint to rh_palm so we can move the hand in space
        # Find rh_palm body
        palm_body = worldbody.find(".//body[@name='rh_palm']")
        if palm_body is not None:
            # Check if freejoint already exists
            if palm_body.find('freejoint') is None:
                # Add freejoint as first child
                freejoint = ET.Element('freejoint', attrib={'name': 'hand_freejoint'})
                palm_body.insert(0, freejoint)
                cprint("[INFO] Added freejoint to rh_palm for 6DOF movement", "cyan")
        
        # Add lighting
        if root.find('visual') is None:
            visual = ET.SubElement(root, 'visual')
            ET.SubElement(visual, 'headlight', attrib={
                'ambient': '0.5 0.5 0.5',
                'diffuse': '0.5 0.5 0.5',
                'specular': '0.3 0.3 0.3'
            })
        
        # Add ground if requested
        if self.show_ground:
            ground = ET.SubElement(worldbody, 'geom', attrib={
                'name': 'ground',
                'type': 'plane',
                'size': '2 2 0.1',
                'rgba': '0.9 0.9 0.9 1',
                'pos': '0 0 0'
            })
        
        # Add table if requested
        if self.show_table:
            table = ET.SubElement(worldbody, 'body', attrib={
                'name': 'table',
                'pos': '0.5 0 -0.025'
            })
            ET.SubElement(table, 'geom', attrib={
                'type': 'box',
                'size': '0.4 0.4 0.025',
                'rgba': '0.8 0.6 0.4 1',
                'mass': '0'
            })
        
        # Ensure asset section exists
        if root.find('asset') is None:
            ET.SubElement(root, 'asset')
        
        return root
    
    def _add_object_to_xml(self, root: ET.Element, mesh_path: str, 
                          scale: np.ndarray, pose: np.ndarray) -> ET.Element:
        """Add object mesh to XML"""
        # Get asset section
        asset = root.find('asset')
        if asset is None:
            asset = ET.SubElement(root, 'asset')
        
        # Add mesh to assets
        mesh_name = "object_mesh"
        ET.SubElement(asset, 'mesh', attrib={
            'name': mesh_name,
            'file': mesh_path,
            'scale': f'{scale[0]} {scale[1]} {scale[2]}'
        })
        
        # Add object body to worldbody
        worldbody = root.find('worldbody')
        
        # Extract position and quaternion from pose
        position = pose[:3]
        quat_wxyz = pose[3:]  # [w, x, y, z]
        
        object_body = ET.SubElement(worldbody, 'body', attrib={
            'name': 'object',
            'pos': f'{position[0]} {position[1]} {position[2]}',
            'quat': f'{quat_wxyz[0]} {quat_wxyz[1]} {quat_wxyz[2]} {quat_wxyz[3]}'
        })
        
        # Add mesh geom
        ET.SubElement(object_body, 'geom', attrib={
            'type': 'mesh',
            'mesh': mesh_name,
            'rgba': '0.7 0.7 0.7 1',
            'contype': '0',
            'conaffinity': '0'
        })
        
        return root
    
    def _build_model(self, with_object: bool = False) -> None:
        """Build MuJoCo model"""
        # Create base XML
        root = self._create_base_xml()
        
        # Add object if needed
        if with_object and self.current_object_mesh is not None:
            root = self._add_object_to_xml(
                root, 
                self.current_object_mesh,
                self.current_object_scale,
                self.current_object_pose
            )
        
        # Write to temporary file
        self.temp_dir = tempfile.mkdtemp(prefix="dexonomy_mujoco_")
        temp_xml = Path(self.temp_dir) / "scene.xml"
        
        tree = ET.ElementTree(root)
        # ET.indent() is only available in Python 3.9+, so we skip it
        # Indentation is not necessary for MuJoCo to parse the XML
        tree.write(temp_xml, encoding='utf-8', xml_declaration=True)
        
        # Load model
        try:
            self.model = mujoco.MjModel.from_xml_path(str(temp_xml))
            self.data = mujoco.MjData(self.model)
            
            cprint(f"[INFO] MuJoCo model loaded successfully", "green")
            cprint(f"  DOF: {self.model.nv}", "cyan")
            cprint(f"  Bodies: {self.model.nbody}", "cyan")
            cprint(f"  Joints: {self.model.njnt}", "cyan")
            
        except Exception as e:
            cprint(f"[ERROR] Failed to load MuJoCo model: {e}", "red")
            raise
    
    def load_object(self, data: Dict):
        """
        Load object from Dexonomy data with correct scale
        
        Args:
            data: Dataset sample dictionary (contains single grasp)
        """
        # Load object mesh
        mesh_path = data["object_mesh_path"]
        object_id = data["object_id"]
        
        # Get object configuration for scale
        scene_config = data["scene_config"]
        obj_config = scene_config[object_id]
        base_scale = obj_config["scale"]
        obj_pose = obj_config["pose"]
        obj_pos = obj_pose[:3]
        obj_quat = obj_pose[3:]
        
        actual_scale = np.array(base_scale)
        
        # Get object pose based on user preference
        if self.use_tabletop_pose:
            # Use pose from tabletop_pose.json (grasp phase = index 1)
            object_poses = data["object_poses"]
            grasp_pose_wxyz = np.array(object_poses[1])
            
            position = grasp_pose_wxyz[:3]
            quat_wxyz = grasp_pose_wxyz[3:]
        else:
            # Use origin pose
            position = obj_pos
            quat_wxyz = obj_quat
        
        # Store object info
        self.current_object_mesh = mesh_path
        self.current_object_scale = actual_scale
        self.current_object_pose = np.concatenate([position, quat_wxyz])
        
        cprint(f"[INFO] Prepared object: {object_id}", "yellow")
        cprint(f"  Object scale: {actual_scale}", "white")
        cprint(f"  Pose source: {'tabletop_pose.json' if self.use_tabletop_pose else 'origin'}", "white")
        cprint(f"  Pose: position={position}, quat_wxyz={quat_wxyz}", "white")
    
    def set_shadow_qpos(self, shadow_qpos: np.ndarray):
        """
        Set Shadow Hand qpos from Dexonomy dataset
        
        Dexonomy 29D format:
        - [0:3]: base translation (x, y, z)
        - [3:7]: base quaternion (w, x, y, z)
        - [7:29]: 22 joint angles (Shadow Hand without wrist)
        
        MuJoCo qpos format (with freejoint):
        - [0:3]: freejoint position (x, y, z)
        - [3:7]: freejoint quaternion (w, x, y, z)
        - [7:29]: 22 joint angles
        
        Args:
            shadow_qpos: (29,) array from Dexonomy dataset
        """
        if self.model is None or self.data is None:
            cprint("[WARNING] Model not initialized", "yellow")
            return
        
        # Parse 29D data
        base_pos = shadow_qpos[0:3]
        base_quat_wxyz = shadow_qpos[3:7]  # Dexonomy uses (w, x, y, z)
        joint_angles = shadow_qpos[7:29]  # 22 joint angles
        
        cprint(f"\n[DEBUG] Setting Shadow Hand pose:", "yellow")
        cprint(f"  Model nq: {self.model.nq}", "white")
        cprint(f"  Base position: {base_pos}", "white")
        cprint(f"  Base quat (wxyz): {base_quat_wxyz}", "white")
        cprint(f"  Joint angles shape: {joint_angles.shape}", "white")
        
        # Check if we have freejoint (qpos should be 29: 7 for freejoint + 22 for joints)
        if self.model.nq == 29:
            # Perfect! We have freejoint + 22 joints
            # Set freejoint position (qpos[0:3])
            self.data.qpos[0:3] = base_pos
            # Set freejoint quaternion (qpos[3:7]) - MuJoCo uses (w, x, y, z)
            self.data.qpos[3:7] = base_quat_wxyz
            # Set joint angles (qpos[7:29])
            self.data.qpos[7:29] = joint_angles
            cprint(f"  ✓ Set full qpos (freejoint + joints)", "green")
        elif self.model.nq == 22:
            # No freejoint, only 22 joints
            self.data.qpos[:] = joint_angles
            cprint(f"  ✓ Set joint angles only (no freejoint)", "green")
        else:
            cprint(f"  ✗ Unexpected nq: {self.model.nq}", "red")
            # Try best effort
            min_len = min(self.model.nq, len(shadow_qpos))
            self.data.qpos[:min_len] = shadow_qpos[:min_len]
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        cprint(f"  ✓ Forward kinematics complete", "green")
    
    def render_grasp_single(self, data: Dict, fps: int = 10):
        """
        Render a single grasp from Dexonomy data
        
        Args:
            data: Dictionary from DexonomyGraspDataset (single grasp)
            fps: Frames per second (for video recording in headless mode)
        """
        # Get grasp data
        grasp_qpos = data["grasp_qpos"]
        object_id = data["object_id"]
        
        # Load object
        self.load_object(data)
        
        # Build MuJoCo model with object
        self._build_model(with_object=True)
        
        # Set grasp pose
        self.set_shadow_qpos(grasp_qpos)
        
        # Display info
        cprint(f"\n[INFO] Rendering grasp:", "green")
        cprint(f"  Grasp Type: {data['grasp_type']}", "yellow")
        cprint(f"  Object ID: {object_id}", "yellow")
        cprint(f"  Scale Name: {data['scale_name']}", "yellow")
        cprint(f"  Grasp Index: {data['grasp_idx']}/{data['num_grasps_in_file']}", "yellow")
        
        if self.headless:
            # Headless rendering for video
            self._render_headless(data, fps)
        else:
            # Interactive viewer
            self._render_interactive()
    
    def _render_interactive(self):
        """Launch interactive MuJoCo viewer"""
        cprint(f"\n[INFO] Starting interactive viewer (press ESC to quit)...", "cyan")
        
        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera position
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.lookat[:] = [0.0, 0.0, 0.2]
            
            # Keep viewer open
            while viewer.is_running():
                # Update simulation (without stepping)
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
        
        cprint(f"\n[INFO] Visualization complete!", "green")
    
    def _render_headless(self, data: Dict, fps: int):
        """Render headless for video recording"""
        cprint(f"\n[INFO] Rendering in headless mode...", "cyan")
        
        # Setup renderer
        width, height = 1920, 1080
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        # Setup camera
        camera = mujoco.MjvCamera()
        camera.distance = 1.0
        camera.azimuth = 90
        camera.elevation = -20
        camera.lookat[:] = [0.0, 0.0, 0.2]
        
        # Setup video writer
        video_path = Path(__file__).parent / f"dexonomy_mujoco_{data['grasp_type']}.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        cprint(f"[INFO] Saving video to {video_path}", "cyan")
        
        # Render frames
        frames_to_render = 30
        for frame in range(frames_to_render):
            # Update simulation
            mujoco.mj_forward(self.model, self.data)
            
            # Render
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render()
            
            # Write frame
            writer.write(rgb[..., ::-1])  # Convert RGB to BGR for OpenCV
        
        writer.release()
        cprint(f"\n[INFO] Video saved successfully to {video_path}!", "green")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            import shutil
            shutil.rmtree(self.temp_dir)


def main():
    """Example usage"""
    from dexonomy_dataset import DexonomyGraspDataset
    
    # Create viewer
    viewer = DexonomyGraspMuJoCoViewer(
        headless=False,
        show_ground=False,
        show_table=True,
        use_tabletop_pose=False,
    )
    
    # Load dataset
    data_root = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")
    dataset = DexonomyGraspDataset(
        data_root=data_root,
        grasp_type="5_Light_Tool",  # 修正：使用 grasp_type（单数）
        split="train",
    )
    
    # Get a sample
    data = dataset[0]
    
    # Render
    try:
        viewer.render_grasp_single(data, fps=10)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

