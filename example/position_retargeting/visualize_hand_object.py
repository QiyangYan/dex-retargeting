from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tyro

from dataset import DexYCBVideoDataset
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from hand_viewer import HandDatasetSAPIENViewer
from hand_robot_viewer_img import RobotHandDatasetSAPIENViewer_IMG

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_



def viz_hand_object(robots: Optional[Tuple[RobotName]], data_root: Path, fps: int, img: bool = False, retargeting_type: str = "POSITION", data_id: int = 8, two_optimizers: bool = False, second_optimizer_type: str = "VECTOR", subject_id: str = "20200709-subject-01"):
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
            list(robots), HandType.right, headless=False, retargeting_type=retargeting_type
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
            list(robots), HandType.right, headless=False, retargeting_type=retargeting_type, two_optimizers=two_optimizers, second_optimizer_type=second_optimizer_type
        )

    # Data ID, feel free to change it to visualize different trajectory
    sampled_data = dataset[data_id]
    for key, value in sampled_data.items():
        if "pose" not in key:
            print(f"{key}: {value}")
    viewer.load_object_hand(sampled_data)
    viewer.render_dexycb_data(sampled_data, fps)


def main(dexycb_dir: str="/home/guizhewei/guizhewei/Dexycb_dataset", robots: Optional[List[RobotName]] = None, fps: int = 10, img: bool = False, retargeting_type: str = "POSITION", data_id: int = 8, two_optimizers: bool = False, second_optimizer_type: str = "VECTOR", subject_id: str = "20200709-subject-01"):
    """
    Render the human and robot trajectories for grasping object inside DexYCB dataset.
    The human trajectory is visualized as provided, while the robot trajectory is generated from retargeting

    Args:
        dexycb_dir: Data root path to the dexycb dataset
        robots: The names of robots to render, if None, render human hand trajectory only
        fps: frequency to render hand-object trajectory
        img: whether to use single frame for retargeting
        retargeting_type: retargeting type, either "POSITION", "VECTOR", "FINGERTIP", or "DEXPILOT"
        data_id: which data to visualize
        two_optimizers: whether to use two optimizers for retargeting
        second_optimizer_type: type of the second optimizer when two_optimizers=True, either "VECTOR", "FINGERTIP", or "DEXPILOT"
        subject_id: Subject ID string (e.g., "20200709-subject-01" or "20200813-subject-02")

    """
    data_root = Path(dexycb_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    if not data_root.exists():
        raise ValueError(f"Path to DexYCB dir: {data_root} does not exist.")
    else:
        print(f"Using DexYCB dir: {data_root}")

    viz_hand_object(robots, data_root, fps, img, retargeting_type, data_id, two_optimizers, second_optimizer_type, subject_id)


if __name__ == "__main__":
    tyro.cli(main)
