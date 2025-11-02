"""
存储 Shadow Hand 到其他机器手的 retargeting 结果
模仿 store_hand_object.py 的逻辑，批量处理 Dexonomy 数据集
"""
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import numpy as np
import tyro
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from termcolor import cprint

from dexonomy_dataset import DexonomyGraspDataset
from dexonomy_viewer import DexonomyGraspSAPIENViewer
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def retarget_shadow_to_robot_no_offset(
    viewer: DexonomyGraspSAPIENViewer,
    shadow_qpos: np.ndarray,
    robot_idx: int,
) -> Optional[np.ndarray]:
    """
    从 Shadow Hand qpos retarget 到目标机器人，返回 qpos（不包含 y_offset）
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        shadow_qpos: (29,) Shadow Hand qpos
        robot_idx: 目标机器人索引（>= 1，0 是 Shadow Hand）
    
    Returns:
        retargeted_qpos: retarget 后的 qpos，如果没有 retargeting 则返回 None
    """
    if robot_idx == 0 or robot_idx >= len(viewer.robots):
        return None
    
    retargeting = viewer.retargetings[robot_idx]
    if retargeting is None:
        return None
    
    # 计算 Shadow Hand 关节位置（使用 FK）
    joint_positions = viewer._compute_shadow_joint_positions(shadow_qpos)
    
    # 根据 retargeting 类型获取参考值（第一个优化器）
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    
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
    
    # 对齐机器人手与 Shadow Hand 的 palm frame
    canonical_frame = retargeting.optimizer.canonical_frame
    shadow_rotation = Rotation.from_quat([shadow_qpos[4], shadow_qpos[5], shadow_qpos[6], shadow_qpos[3]])
    robot_r = shadow_rotation * Rotation.from_matrix(canonical_frame)
    
    # 获取完整的 retargeting 输出（第一个优化器）
    last_pos = np.concatenate([shadow_qpos[:3], robot_r.as_euler("XYZ", degrees=False), retargeting.mean_qpos[6:]])
    qpos_full = retargeting.retarget(ref_value, last_qpos=last_pos)
    qpos = qpos_full[viewer.retarget2sapien[robot_idx]]
    
    # 第二个优化器用于手指细化
    if viewer.two_optimizers and viewer.second_retargeting is not None:
        second_retargeting_type = viewer.second_retargeting.optimizer.retargeting_type
        second_indices = viewer.second_retargeting.optimizer.target_link_human_indices
        
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
        
        # 使用第一个优化器的结果更新第二个优化器的 last_qpos
        viewer.second_retargeting.last_qpos = qpos_full[viewer.second_retargeting.optimizer.idx_pin2target]
        
        # 使用第二个优化器进行 retargeting
        qpos_second = viewer.second_retargeting.retarget(second_ref_value)[viewer.retarget2sapien[robot_idx]]
        qpos = qpos_second  # 使用第二个优化器的结果
    
    # 注意：这里不应用 y_offset，直接返回 qpos
    return qpos.copy()


def store_retargeted_poses(
    robots: List[RobotName],
    data_root: Path,
    retargeting_type: str = "VECTOR",
    two_optimizers: bool = False,
    second_optimizer_type: str = "FINGERTIP",
    grasp_type: Optional[str] = None,
    object_ids: Optional[List[str]] = None,
    split: str = "train",
    data_id_start: Optional[int] = None,
    data_id_end: Optional[int] = None,
    visualize: bool = False,
):
    """
    批量处理 Dexonomy 数据集，存储 Shadow Hand 到其他机器手的 retargeting 结果
    
    Args:
        robots: 目标机器人列表（第一个应该是 Shadow Hand，后续是需要 retarget 的机器人）
        data_root: Dexonomy 数据集根目录
        retargeting_type: retargeting 类型，"POSITION", "VECTOR", "FINGERTIP", 或 "DEXPILOT"
        two_optimizers: 是否使用两个优化器
        second_optimizer_type: 第二个优化器类型
        grasp_type: 特定抓取类型（None 表示所有类型）
        object_ids: 特定物体 ID 列表（None 表示所有物体）
        split: 数据集划分 "train", "test", 或 "all"
        data_id_start: 起始数据索引（None 表示从头开始）
        data_id_end: 结束数据索引（None 表示处理到最后）
        visualize: 是否可视化（False 表示 headless 模式）
    """
    # 转换 retargeting_type
    if retargeting_type == "POSITION":
        retargeting_type_enum = RetargetingType.position
    elif retargeting_type == "VECTOR":
        retargeting_type_enum = RetargetingType.vector
    elif retargeting_type == "FINGERTIP":
        retargeting_type_enum = RetargetingType.fingertip
    elif retargeting_type == "DEXPILOT":
        retargeting_type_enum = RetargetingType.dexpilot
    else:
        raise ValueError(f"Unsupported retargeting type: {retargeting_type}")
    
    # 加载数据集
    cprint(f"[INFO] Loading Dexonomy dataset...", "green")
    dataset = DexonomyGraspDataset(
        data_root=data_root,
        grasp_type=grasp_type,
        object_ids=object_ids,
        split=split,
    )
    
    if len(dataset) == 0:
        cprint(f"[ERROR] Dataset is empty!", "red")
        return
    
    cprint(f"[INFO] Loaded {len(dataset)} grasp samples", "green")
    
    # 创建 viewer（headless 模式用于批量处理）
    headless = not visualize
    cprint(f"[INFO] Creating viewer (headless={headless})...", "green")
    viewer = DexonomyGraspSAPIENViewer(
        robot_names=robots,
        headless=headless,
        data_root=data_root,
        hand_type=HandType.right,
        retargeting_type=retargeting_type_enum,
        two_optimizers=two_optimizers,
        second_optimizer_type=second_optimizer_type,
    )
    
    # 确定处理范围
    data_id_start = data_id_start if data_id_start is not None else 0
    data_id_end = data_id_end if data_id_end is not None else len(dataset)
    data_indices = range(data_id_start, min(data_id_end, len(dataset)))
    
    cprint(f"[INFO] Processing data indices {data_id_start} to {data_id_end-1} (total {len(data_indices)} samples)", "green")
    
    # 存储 retargeted poses
    # 格式: {
    #   data_idx: {
    #     "object_id": str,           # 物体 ID（用于批量加载和标记）
    #     "grasp_type": str,          # 抓取类型
    #     "scale_name": str,          # 缩放名称
    #     "grasp_idx": int,           # 在当前文件中的抓取索引
    #     "scene_scale": float,       # 场景缩放
    #     "shadow_qpos": np.ndarray,  # Shadow Hand 原始 qpos (29,)
    #     "robot_poses": {            # 各机器人的 retargeted qpos（不包含 y_offset）
    #       str(robot_name): np.ndarray
    #     }
    #   }
    # }
    retargeted_poses_dict = defaultdict(dict)
    
    # 按 object_id 分组处理（用于批量加载和标记）
    object_id_to_indices = defaultdict(list)
    for idx in data_indices:
        # 快速获取 object_id（通过索引条目）
        item = dataset.grasp_index[idx]
        object_id = item["object_id"]
        object_id_to_indices[object_id].append(idx)
    
    cprint(f"[INFO] Grouped into {len(object_id_to_indices)} unique objects", "green")
    cprint(f"[INFO] Object distribution: {[(obj_id, len(indices)) for obj_id, indices in list(object_id_to_indices.items())[:5]]}...", "white")
    
    # 批量处理每个 object_id
    for object_id, indices in tqdm(object_id_to_indices.items(), desc="Processing objects"):
        cprint(f"\n[INFO] Processing object {object_id} ({len(indices)} grasps)...", "cyan")
        
        for idx in tqdm(indices, desc=f"Object {object_id}", leave=False):
            try:
                # 加载数据样本
                data = dataset[idx]
                
                # 获取 Shadow Hand qpos（使用 pin_order 版本）
                shadow_qpos = data["grasp_qpos_pin_order"]  # (29,)
                
                # 存储基本信息
                retargeted_poses_dict[idx] = {
                    "object_id": data["object_id"],
                    "grasp_type": data["grasp_type"],
                    "scale_name": data["scale_name"],
                    "grasp_idx": data["grasp_idx"],
                    "scene_scale": data["scene_scale"],
                    "shadow_qpos": shadow_qpos.copy(),  # Shadow Hand 原始 qpos
                    "robot_poses": {},  # 存储每个机器人的 retargeted qpos
                }
                
                # 对每个目标机器人进行 retargeting
                for robot_idx in range(1, len(robots)):  # 跳过 Shadow Hand (idx=0)
                    robot_name = robots[robot_idx]
                    
                    # 执行 retargeting（不包含 y_offset）
                    retargeted_qpos = retarget_shadow_to_robot_no_offset(
                        viewer, shadow_qpos, robot_idx
                    )
                    
                    if retargeted_qpos is not None:
                        retargeted_poses_dict[idx]["robot_poses"][str(robot_name)] = retargeted_qpos.copy()
                        cprint(f"  [DEBUG] Retargeted to {robot_name}: qpos shape {retargeted_qpos.shape}", "white")
                    else:
                        cprint(f"  [WARNING] Failed to retarget to {robot_name}", "yellow")
                
                # 如果可视化，加载物体和手并渲染一帧
                if visualize:
                    viewer.load_object(data)
                    viewer.set_shadow_qpos(shadow_qpos, robot_idx=0, y_offset=0.0)
                    for robot_idx in range(1, len(robots)):
                        retargeted_qpos = retargeted_poses_dict[idx]["robot_poses"].get(str(robots[robot_idx]))
                        if retargeted_qpos is not None:
                            viewer.robots[robot_idx].set_qpos(retargeted_qpos.astype(np.float32))
                    viewer.scene.update_render()
                    if not viewer.headless:
                        viewer.viewer.render()
                    
            except Exception as e:
                cprint(f"[ERROR] Failed to process data {idx}: {e}", "red")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    if len(retargeted_poses_dict) > 0:
        save_name = input("\nEnter the name for the file (without extension): ")
        save_path = data_root / f"grasp_poses_retargeted_{save_name}.npy"
        
        # 转换为普通字典（移除 defaultdict）
        result_dict = dict(retargeted_poses_dict)
        
        np.save(save_path, result_dict)
        cprint(f"\n[INFO] Saved {len(result_dict)} retargeted poses to {save_path}", "green")
        cprint(f"[INFO] Robot names: {[str(r) for r in robots]}", "cyan")
        cprint(f"[INFO] Retargeting type: {retargeting_type}", "cyan")
        if two_optimizers:
            cprint(f"[INFO] Second optimizer type: {second_optimizer_type}", "cyan")
    else:
        cprint("[WARNING] No data to save!", "yellow")


def main(
    dexonomy_dir: str = "/home/guizhewei/guizhewei/Dexonomy_dataset",
    robots: List[RobotName] = [RobotName.shadow_no_wrist, RobotName.allegro],
    retargeting_type: str = "VECTOR",
    two_optimizers: bool = False,
    second_optimizer_type: str = "FINGERTIP",
    grasp_type: Optional[str] = None,
    object_ids: Optional[List[str]] = None,
    split: str = "train",
    data_id_start: Optional[int] = None,
    data_id_end: Optional[int] = None,
    visualize: bool = False,
):
    """
    主函数：存储 Shadow Hand 到其他机器手的 retargeting 结果
    
    Args:
        dexonomy_dir: Dexonomy 数据集根目录
        robots: 机器人列表，第一个必须是 Shadow Hand，后续是需要 retarget 的目标机器人
        retargeting_type: retargeting 类型
        two_optimizers: 是否使用两个优化器
        second_optimizer_type: 第二个优化器类型
        grasp_type: 特定抓取类型
        object_ids: 特定物体 ID 列表
        split: 数据集划分
        data_id_start: 起始数据索引
        data_id_end: 结束数据索引
        visualize: 是否可视化
    """
    data_root = Path(dexonomy_dir).absolute()
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    
    RetargetingConfig.set_default_urdf_dir(robot_dir)
    
    if not data_root.exists():
        raise ValueError(f"Path to Dexonomy dir: {data_root} does not exist.")
    else:
        cprint(f"Using Dexonomy dir: {data_root}", "green")
    
    # 验证第一个机器人是 Shadow Hand
    if robots[0] != RobotName.shadow_no_wrist:
        cprint(f"[WARNING] First robot should be Shadow Hand, but got {robots[0]}", "yellow")
    
    store_retargeted_poses(
        robots=robots,
        data_root=data_root,
        retargeting_type=retargeting_type,
        two_optimizers=two_optimizers,
        second_optimizer_type=second_optimizer_type,
        grasp_type=grasp_type,
        object_ids=object_ids,
        split=split,
        data_id_start=data_id_start,
        data_id_end=data_id_end,
        visualize=visualize,
    )


if __name__ == "__main__":
    tyro.cli(main)

