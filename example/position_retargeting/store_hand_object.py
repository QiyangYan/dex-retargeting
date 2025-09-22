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


def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int, img: bool = False, retargeting_type: str = "POSITION", save_grasp_pose: bool = False):
    if save_grasp_pose:
        headless = True
    else:
        headless = False

    dataset = DexYCBVideoDataset(data_root, hand_type="right")
    if robots is None:
        viewer = HandDatasetSAPIENViewer(headless=False)
    elif img:
        if retargeting_type not in ["POSITION", "VECTOR", "DEXPILOT"]:
            raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
        if retargeting_type == "POSITION":
            retargeting_type = RetargetingType.position
        elif retargeting_type == "VECTOR":
            retargeting_type = RetargetingType.vector
        elif retargeting_type == "DEXPILOT":
            retargeting_type = RetargetingType.dexpilot
        viewer = RobotHandDatasetSAPIENViewer_IMG(
            list(robots), HandType.right, headless=headless, retargeting_type=retargeting_type
        )
    else:
        if retargeting_type not in ["POSITION", "VECTOR", "DEXPILOT"]:
            raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
        if retargeting_type == "POSITION":
            retargeting_type = RetargetingType.position
        elif retargeting_type == "VECTOR":
            retargeting_type = RetargetingType.vector
        elif retargeting_type == "DEXPILOT":
            retargeting_type = RetargetingType.dexpilot
        viewer = RobotHandDatasetSAPIENViewer(
            list(robots), HandType.right, headless=headless, retargeting_type=retargeting_type
        )
    
    grasp_pose_dict = defaultdict(dict)
    for i, sampled_data in enumerate(dataset):
        if i == 45:
            for key, value in sampled_data.items():
                if "pose" not in key:
                    print(f"{key}: {value}")
            data = copy.deepcopy(sampled_data)   # <- important
            if not img:
                viewer.reset_env()                   # <- add (see below)
            viewer.load_object_hand(data)
            if save_grasp_pose: 
                grasp_pose = viewer.render_dexycb_data(sampled_data, fps)
                grasp_pose_dict[i] = grasp_pose
            else:
                viewer.render_dexycb_data(sampled_data, fps)

            break

    # save dict in npy formate
    if save_grasp_pose:
        name = input("Enter the name for the grasp pose file (without extension): ")
        grasp_pose_path = data_root / f"grasp_poses_{name}.npy"
        np.save(grasp_pose_path, dict(grasp_pose_dict))
        print(f"Grasp poses saved to {grasp_pose_path}")


def main(dexycb_dir: str="/home/ubuntu/Documents/DexYCB", robots: Optional[List[RobotName]] = None, fps: int = 10, img: bool = False, retargeting_type: str = "POSITION", save_grasp_pose: bool = False):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from position retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory

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

    viz_hand_object(robots, data_root, fps, img, retargeting_type, save_grasp_pose)


if __name__ == "__main__":
    tyro.cli(main)
