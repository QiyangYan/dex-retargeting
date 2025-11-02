"""
Dataset loader for Dexonomy grasp dataset
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class DexonomyGraspDataset:
    """Load Dexonomy grasp dataset with Shadow Hand qpos and object information"""
    
    # TODO:31 grasp types from GRASP taxonomy
    GRASP_TYPES = [
        # "1_Large_Diameter",
        # "2_Small_Diameter", 
        # "3_Medium_Wrap",
        # "4_Adducted_Thumb",
        "5_Light_Tool",
        "6_Prismatic_4_Finger",
        "7_Prismatic_3_Finger",
        "8_Prismatic_2_Finger",
        "9_Palmar_Pinch",
        "10_Power_Disk",
        "11_Power_Sphere",
        "12_Precision_Disk",
        "13_Precision_Sphere",
        "14_Tripod",
        "15_Fixed_Hook",
        "16_Lateral",
        "17_Index_Finger_Extension",
        "18_Extensior_Type",
        "20_Writing_Tripod",
        "22_Parallel_Extension",
        "23_Adduction_Grip",
        "24_Tip_Pinch",
        "25_Lateral_Tripod",
        "26_Sphere_4_Finger",
        "27_Quadpod",
        "28_Sphere_3_Finger",
        "29_Stick",
        "30_Palmar",
        "31_Ring",
        "32_Ventral",
        "33_Inferior_Pincer",
    ]
    
    def __init__(
        self, 
        data_root: Path,
        grasp_type: Optional[str] = None,
        object_ids: Optional[List[str]] = None,
        split: str = "train",
    ):
        """
        Args:
            data_root: Root directory of Dexonomy dataset
            grasp_type: Specific grasp type to load (e.g., "1_Large_Diameter")
            object_ids: List of specific object IDs to load
            split: "train", "test", or "all"
        """
        self.data_root = Path(data_root)
        self.grasp_dir = self.data_root / "Dexonomy_GRASP_shadow" / "succ_collect"
        self.object_dir = self.data_root / "objaverse_5k" / "processed_data"
        self.scene_cfg_dir = self.data_root / "objaverse_5k" / "scene_cfg"
        self.split_dir = self.data_root / "objaverse_5k" / "valid_split"
        
        # Predefine the joint order
        # FMRLT
        bodex_order = [
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
            "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
            "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
        ]
        sapien_order = [
            'rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_LFJ5', 'rh_THJ5', 
            'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_LFJ4', 'rh_THJ4', 
            'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_LFJ3', 'rh_THJ3', 
            'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_LFJ2', 'rh_THJ2', 
            'rh_LFJ1', 'rh_THJ1']
        
        #FLMRT
        pinocchio_order = [
            "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
            "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
            "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
            "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
            "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
        ]
        
        
        # TODO:for tm_viewer
        # sapien_order = [
        #     "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
        #     "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
        #     "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
        #     "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
        #     "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
        # ]
        self.bodex_order = bodex_order
        self.sapien_order = sapien_order
        self.pinocchio_order = pinocchio_order

        self.bodex_to_sapien_order_list = list(range(7)) + [bodex_order.index(joint) + 7 for joint in sapien_order]
        self.sapien_to_bodex_order_list = list(range(7)) + [sapien_order.index(joint) + 7 for joint in bodex_order]
        self.bodex_to_pinocchio_order_list = list(range(7)) + [bodex_order.index(joint) + 7 for joint in pinocchio_order]
        # Load split
        split_file = self.split_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file, "r") as f:
                self.valid_object_ids = set(json.load(f))
        else:
            self.valid_object_ids = None
        
        # Build index of all available grasps
        self.grasp_index = []
        self._build_grasp_index(grasp_type, object_ids)
        
        print(f"[INFO] Loaded {len(self.grasp_index)} grasp samples from Dexonomy dataset")
    
    def _build_grasp_index(
        self, 
        grasp_type: Optional[str] = None,
        object_ids: Optional[List[str]] = None
    ):
        """Build index of all available grasp samples (each grasp individually)"""
        grasp_types = [grasp_type] if grasp_type else self.GRASP_TYPES
        #TODO: if tackle in batches ,remove this
        max_iters = 10000

        for gtype in grasp_types:
            grasp_type_dir = self.grasp_dir / gtype
            if not grasp_type_dir.exists():
                continue
            
            # Iterate through object IDs
            for object_path in grasp_type_dir.iterdir():
                if not object_path.is_dir():
                    continue
                
                object_id = object_path.name
                
                # Filter by split
                if self.valid_object_ids and object_id not in self.valid_object_ids:
                    continue
                
                # Filter by specific object IDs
                if object_ids and object_id not in object_ids:
                    continue
                
                # Check if object data exists
                obj_data_path = self.object_dir / object_id
                if not obj_data_path.exists():
                    continue
                
                # Iterate through scene types (floating)
                scene_type_dir = object_path / "floating"
                if not scene_type_dir.exists():
                    continue
                
                # Iterate through different scales
                for scale_file in scene_type_dir.glob("scale*.npy"):
                    scale_name = scale_file.stem
                    
                    # Check if scene config exists
                    scene_cfg_file = self.scene_cfg_dir / object_id / "floating" / f"{scale_name}.npy"
                    if not scene_cfg_file.exists():
                        continue
                    
                    # Load the grasp file to get number of grasps
                    try:
                        grasp_data = np.load(scale_file, allow_pickle=True).item()
                        num_grasps = len(grasp_data["grasp_qpos"])
                        
                        # Add each individual grasp as a separate sample
                        for grasp_idx in range(num_grasps):
                            self.grasp_index.append({
                                "grasp_type": gtype,
                                "object_id": object_id,
                                "scale_name": scale_name,
                                "grasp_file": scale_file,
                                "scene_cfg_file": scene_cfg_file,
                                "object_dir": obj_data_path,
                                "grasp_idx": grasp_idx,  # Index within this scale file
                                "num_grasps": num_grasps,  # Total grasps in this file
                            })
                    except Exception as e:
                        print(f"[WARNING] Failed to load {scale_file}: {e}")
                        continue
                if len(self.grasp_index) >= max_iters:
                    break
    def __len__(self):
        return len(self.grasp_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Load a single grasp sample with all necessary information
        
        Returns:
            Dict containing:
                - grasp_type: str
                - object_id: str
                - scale_name: str
                - grasp_idx: int (index within the scale file)
                - grasp_qpos: (29,) array of Shadow Hand joint positions for THIS grasp
                - pregrasp_qpos: (29,) array
                - squeeze_qpos: (29,) array
                - scene_scale: float (scale factor for THIS grasp)
                - scene_config: dict with object pose, scale, paths
                - object_mesh_path: path to simplified.obj
                - object_urdf_path: path to coacd.urdf
                - object_info: dict with mass, obb, etc.
        """
        item = self.grasp_index[idx]
        grasp_idx = item["grasp_idx"]
        
        # Load grasp qpos data
        grasp_data = np.load(item["grasp_file"], allow_pickle=True).item()
        
        # Load scene configuration
        scene_cfg = np.load(item["scene_cfg_file"], allow_pickle=True).item()
        
        # Load object info
        object_info_file = item["object_dir"] / "info" / "simplified.json"
        with open(object_info_file, "r") as f:
            object_info = json.load(f)
        
        # Load object poses from tabletop_pose.json
        tabletop_pose_file = item["object_dir"] / "info" / "tabletop_pose.json"
        with open(tabletop_pose_file, "r") as f:
            tabletop_poses = json.load(f)  # List of [x, y, z, qx, qy, qz, qw]
        
        # Get mesh and urdf paths
        object_mesh_path = item["object_dir"] / "mesh" / "simplified.obj"
        object_urdf_path = item["object_dir"] / "urdf" / "coacd.urdf"

        # Extract THIS specific grasp (single grasp, not all grasps)
        return {
            "grasp_type": item["grasp_type"],
            "object_id": item["object_id"],
            "scale_name": item["scale_name"],
            "grasp_idx": grasp_idx,
            "num_grasps_in_file": item["num_grasps"],
            
            # Single grasp data (29,) instead of (N, 29)
            "grasp_qpos": grasp_data["grasp_qpos"][grasp_idx][self.bodex_to_sapien_order_list],  # (29,)
            "pregrasp_qpos": grasp_data["pregrasp_qpos"][grasp_idx][self.bodex_to_sapien_order_list],  # (29,)
            "squeeze_qpos": grasp_data["squeeze_qpos"][grasp_idx][self.bodex_to_sapien_order_list],  # (29,)
            "scene_scale": float(grasp_data["scene_scale"][grasp_idx]),  # scalar

            # grasp pose for pinocchio
            "grasp_qpos_pin_order": grasp_data["grasp_qpos"][grasp_idx][self.bodex_to_pinocchio_order_list],
            "pregrasp_qpos_pin_order": grasp_data["pregrasp_qpos"][grasp_idx][self.bodex_to_pinocchio_order_list],
            "squeeze_qpos_pin_order": grasp_data["squeeze_qpos"][grasp_idx][self.bodex_to_pinocchio_order_list],
            
            # Object poses for different phases (from tabletop_pose.json)
            # Format: [x, y, z, qw, qx, qy, qz] - quaternion in wxyz format
            "object_poses": tabletop_poses,  # List of poses for [pregrasp, grasp, squeeze, ...]
            
            "scene_config": scene_cfg["scene"],
            "scene_id": scene_cfg["scene_id"],
            "task": scene_cfg["task"],
            "object_mesh_path": str(object_mesh_path),
            "object_urdf_path": str(object_urdf_path),
            "object_info": object_info,
        }
    
    def get_grasp_by_object_and_type(
        self, 
        object_id: str, 
        grasp_type: str,
        scale_name: str = "scale005"
    ) -> Optional[Dict]:
        """Get a specific grasp by object ID, grasp type, and scale"""
        for idx, item in enumerate(self.grasp_index):
            if (item["object_id"] == object_id and 
                item["grasp_type"] == grasp_type and
                item["scale_name"] == scale_name):
                return self[idx]
        return None
    
    def get_all_grasps_for_object(self, object_id: str) -> List[Dict]:
        """Get all grasps for a specific object across all grasp types"""
        grasps = []
        for idx, item in enumerate(self.grasp_index):
            if item["object_id"] == object_id:
                grasps.append(self[idx])
        return grasps
    
    def get_random_grasp(self) -> Dict:
        """Get a random grasp sample"""
        idx = np.random.randint(0, len(self))
        return self[idx]

