"""
Dataset loader for BoDex grasp dataset (Shadow Hand grasps on DGN_2k objects)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class BoDexGraspDataset:
    """Load BoDex grasp dataset with Shadow Hand qpos and object information."""

    def __init__(
        self,
        data_root: Path,
        object_ids: Optional[List[str]] = None,
        split: str = "train",
        env_types: Optional[List[str]] = None,
    ):
        """
        Args:
            data_root: Root directory of BoDex dataset (contains bodex_shadow/ and DGN_2k/)
            object_ids: Filter to specific object IDs
            split: "train", "test", or "all" (uses DGN_2k/valid_split/*.json)
            env_types: Filter to specific environment folders (e.g. ["floating"])
        """
        self.data_root = Path(data_root)
        self.grasp_dir = self.data_root / "bodex_shadow" / "succ_collect"
        self.object_dir = self.data_root / "DGN_2k" / "processed_data"
        self.scene_cfg_dir = self.data_root / "DGN_2k" / "scene_cfg"
        self.split_dir = self.data_root / "DGN_2k" / "valid_split"

        # Joint order definitions (same as Dexonomy dataset)
        bodex_order = [
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",
            "rh_FFJ1",
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",
            "rh_MFJ1",
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",
            "rh_RFJ1",
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",
            "rh_LFJ1",
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",
            "rh_THJ1",
        ]
        sapien_order = [
            "rh_FFJ4",
            "rh_MFJ4",
            "rh_RFJ4",
            "rh_LFJ5",
            "rh_THJ5",
            "rh_FFJ3",
            "rh_MFJ3",
            "rh_RFJ3",
            "rh_LFJ4",
            "rh_THJ4",
            "rh_FFJ2",
            "rh_MFJ2",
            "rh_RFJ2",
            "rh_LFJ3",
            "rh_THJ3",
            "rh_FFJ1",
            "rh_MFJ1",
            "rh_RFJ1",
            "rh_LFJ2",
            "rh_THJ2",
            "rh_LFJ1",
            "rh_THJ1",
        ]
        pinocchio_order = [
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",
            "rh_FFJ1",
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",
            "rh_LFJ1",
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",
            "rh_MFJ1",
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",
            "rh_RFJ1",
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",
            "rh_THJ1",
        ]
        pk_order = [
            "rh_FFJ4",
            "rh_FFJ3",
            "rh_FFJ2",
            "rh_FFJ1",
            "rh_MFJ4",
            "rh_MFJ3",
            "rh_MFJ2",
            "rh_MFJ1",
            "rh_RFJ4",
            "rh_RFJ3",
            "rh_RFJ2",
            "rh_RFJ1",
            "rh_LFJ5",
            "rh_LFJ4",
            "rh_LFJ3",
            "rh_LFJ2",
            "rh_LFJ1",
            "rh_THJ5",
            "rh_THJ4",
            "rh_THJ3",
            "rh_THJ2",
            "rh_THJ1",
        ]

        self.bodex_order = bodex_order
        self.sapien_order = sapien_order
        self.pinocchio_order = pinocchio_order
        self.pk_order = pk_order

        self.bodex_to_sapien_order_list = list(range(7)) + [
            bodex_order.index(joint) + 7 for joint in sapien_order
        ]
        self.sapien_to_bodex_order_list = list(range(7)) + [
            sapien_order.index(joint) + 7 for joint in bodex_order
        ]
        self.bodex_to_pinocchio_order_list = list(range(7)) + [
            bodex_order.index(joint) + 7 for joint in pinocchio_order
        ]
        self.bodex_to_pk_order_list = list(range(7)) + [
            bodex_order.index(joint) + 7 for joint in pk_order
        ]

        # Load split information if available
        self.valid_object_ids = None
        if split != "all":
            split_file = self.split_dir / f"{split}.json"
            if split_file.exists():
                with open(split_file, "r") as f:
                    self.valid_object_ids = set(json.load(f))

        self.object_filter = set(object_ids) if object_ids else None
        self.env_types = set(env_types) if env_types else None

        # Build grasp index
        self.grasp_index: List[Dict] = []
        self._build_grasp_index()

        print(f"[INFO] Loaded {len(self.grasp_index)} grasp samples from BoDex dataset")

    def _build_grasp_index(self) -> None:
        if not self.grasp_dir.exists():
            raise FileNotFoundError(f"Grasp directory not found: {self.grasp_dir}")

        for object_path in sorted(self.grasp_dir.iterdir()):
            if not object_path.is_dir():
                continue

            object_id = object_path.name

            if self.valid_object_ids and object_id not in self.valid_object_ids:
                continue
            if self.object_filter and object_id not in self.object_filter:
                continue

            object_processed_dir = self.object_dir / object_id
            if not object_processed_dir.exists():
                continue

            for env_dir in sorted(object_path.iterdir()):
                if not env_dir.is_dir():
                    continue
                env_name = env_dir.name
                if self.env_types and env_name not in self.env_types:
                    continue

                for scale_file in sorted(env_dir.glob("scale*.npy")):
                    try:
                        grasp_data = np.load(scale_file, allow_pickle=True).item()
                    except Exception as exc:
                        print(f"[WARNING] Failed to load {scale_file}: {exc}")
                        continue

                    if "grasp_qpos" not in grasp_data:
                        continue

                    scene_cfg_file = self._resolve_scene_cfg_path(
                        grasp_data.get("scene_path"), object_id, env_name, scale_file
                    )
                    if scene_cfg_file is None or not scene_cfg_file.exists():
                        continue

                    num_grasps = int(grasp_data["grasp_qpos"].shape[0])
                    for grasp_idx in range(num_grasps):
                        self.grasp_index.append(
                            {
                                "object_id": object_id,
                                "env": env_name,
                                "scale_name": scale_file.stem,
                                "grasp_file": scale_file,
                                "scene_cfg_file": scene_cfg_file,
                                "object_dir": object_processed_dir,
                                "grasp_idx": grasp_idx,
                                "num_grasps": num_grasps,
                            }
                        )

    def _resolve_scene_cfg_path(
        self,
        scene_path_str: Optional[str],
        object_id: str,
        env_name: str,
        scale_file: Path,
    ) -> Optional[Path]:
        if not scene_path_str:
            return None

        scene_path = Path(scene_path_str)
        if "DGN_2k" in scene_path.parts:
            idx = scene_path.parts.index("DGN_2k")
            scene_rel = Path(*scene_path.parts[idx:])
            candidate = self.data_root / scene_rel
            if candidate.exists():
                return candidate

        # Fallback: use standard scene_cfg directory
        scene_cfg_candidate = (
            self.scene_cfg_dir / object_id / env_name / f"{scale_file.stem}.npy"
        )
        if scene_cfg_candidate.exists():
            return scene_cfg_candidate

        print(
            f"[WARNING] Scene config not found for {scale_file}, tried {scene_path_str}"
        )
        return None

    def __len__(self) -> int:
        return len(self.grasp_index)

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.grasp_index):
            raise IndexError(f"Index {idx} out of range for grasp dataset")

        item = self.grasp_index[idx]
        grasp_data = np.load(item["grasp_file"], allow_pickle=True).item()
        scene_cfg = np.load(item["scene_cfg_file"], allow_pickle=True).item()

        object_dir: Path = item["object_dir"]
        object_info_file = object_dir / "info" / "simplified.json"
        with open(object_info_file, "r") as f:
            object_info = json.load(f)

        tabletop_pose_file = object_dir / "info" / "tabletop_pose.json"
        if tabletop_pose_file.exists():
            with open(tabletop_pose_file, "r") as f:
                tabletop_poses = json.load(f)
        else:
            tabletop_poses = []

        scene_entry = self._extract_scene_entry(scene_cfg, item["object_id"])
        obj_scale = np.array(scene_entry.get("scale", np.ones(3)), dtype=float)
        scene_scale = 1.0

        return {
            "object_id": item["object_id"],
            "env": item["env"],
            "scale_name": item["scale_name"],
            "grasp_idx": item["grasp_idx"],
            "num_grasps_in_file": item["num_grasps"],
            "grasp_type": scene_cfg.get("task", {}).get("type", item["env"]),
            "scene_id": scene_cfg.get("scene_id"),
            "task": scene_cfg.get("task", {}),
            "scene_scale": scene_scale,
            "obj_scale": obj_scale,
            "scene_config": scene_cfg["scene"],
            "object_mesh_path": str(object_dir / "mesh" / "simplified.obj"),
            "object_urdf_path": str(object_dir / "urdf" / "coacd.urdf"),
            "object_info": object_info,
            "object_poses": tabletop_poses,
            "grasp_qpos": grasp_data["grasp_qpos"][item["grasp_idx"]][
                self.bodex_to_sapien_order_list
            ],
            "pregrasp_qpos": grasp_data["pregrasp_qpos"][item["grasp_idx"]][
                self.bodex_to_sapien_order_list
            ],
            "squeeze_qpos": grasp_data["squeeze_qpos"][item["grasp_idx"]][
                self.bodex_to_sapien_order_list
            ],
            "grasp_qpos_pin_order": grasp_data["grasp_qpos"][item["grasp_idx"]][
                self.bodex_to_pinocchio_order_list
            ],
            "pregrasp_qpos_pin_order": grasp_data["pregrasp_qpos"][item["grasp_idx"]][
                self.bodex_to_pinocchio_order_list
            ],
            "squeeze_qpos_pin_order": grasp_data["squeeze_qpos"][item["grasp_idx"]][
                self.bodex_to_pinocchio_order_list
            ],
            "grasp_qpos_bodex_order": grasp_data["grasp_qpos"][item["grasp_idx"]],
            "pregrasp_qpos_bodex_order": grasp_data["pregrasp_qpos"][item["grasp_idx"]],
            "squeeze_qpos_bodex_order": grasp_data["squeeze_qpos"][item["grasp_idx"]],
            "grasp_pos_pk_order": grasp_data["grasp_qpos"][item["grasp_idx"]][
                self.bodex_to_pk_order_list
            ],
            "pregrasp_pos_pk_order": grasp_data["pregrasp_qpos"][item["grasp_idx"]][
                self.bodex_to_pk_order_list
            ],
            "squeeze_pos_pk_order": grasp_data["squeeze_qpos"][item["grasp_idx"]][
                self.bodex_to_pk_order_list
            ],
        }

    @staticmethod
    def _extract_scene_entry(scene_cfg: Dict, object_id: str) -> Dict:
        scene_dict = scene_cfg.get("scene", {})
        if object_id in scene_dict:
            return scene_dict[object_id]
        if scene_dict:
            # Return first entry if key mismatch (robustness for naming differences)
            first_key = next(iter(scene_dict.keys()))
            return scene_dict[first_key]
        return {}

    def get_all_grasps_for_object(self, object_id: str) -> List[Dict]:
        return [
            self[idx]
            for idx, item in enumerate(self.grasp_index)
            if item["object_id"] == object_id
        ]

    def get_random_grasp(self) -> Dict:
        idx = np.random.randint(0, len(self))
        return self[idx]


