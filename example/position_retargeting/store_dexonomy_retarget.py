"""
存储 Shadow Hand 到其他机器手的 retargeting 结果
模仿 store_hand_object.py 的逻辑，批量处理 Dexonomy 数据集
"""
from pathlib import Path
from typing import Any, Optional, List, Dict
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

import sapien
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
    
    # 处理 canonical_frame 可能为 None 的情况
    if canonical_frame is None or (isinstance(canonical_frame, np.ndarray) and canonical_frame.size == 0):
        # 如果没有 canonical_frame，使用单位矩阵（不进行额外的旋转对齐）
        # 使用单位四元数 [0, 0, 0, 1] 表示无旋转
        canonical_rotation = Rotation.from_quat([0, 0, 0, 1])
    else:
        # 确保 canonical_frame 是正确的形状 (3, 3)
        canonical_frame = np.array(canonical_frame)
        if canonical_frame.shape != (3, 3):
            cprint(f"[WARNING] canonical_frame has unexpected shape {canonical_frame.shape}, using identity", "yellow")
            canonical_rotation = Rotation.from_quat([0, 0, 0, 1])
        else:
            canonical_rotation = Rotation.from_matrix(canonical_frame)
    
    robot_r = shadow_rotation * canonical_rotation
    
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
    max_objects: Optional[int] = None,
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
        max_objects: 最大处理的物体数量（None 表示处理所有物体，1 表示只处理第一个物体）
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
    #   continuous_idx: {            # 连续索引（从 0 开始），作为字典键
    #     "original_idx": int,       # 原始数据集索引（用于回溯）
    #     "object_id": str,           # 物体 ID（用于批量加载和标记）
    #     "grasp_type": str,          # 抓取类型
    #     "scale_name": str,          # 缩放名称
    #     "grasp_idx": int,           # 在当前文件中的抓取索引
    #     "scene_scale": float,       # 场景缩放因子（标量）
    #     "obj_scale": np.ndarray,    # 物体基础缩放 (3,) [sx, sy, sz]
    #     "shadow_qpos": np.ndarray,  # Shadow Hand 原始 qpos (29,)
    #     "robot_pose": [            # Omni Hand 的 retargeted qpos 列表（不包含 y_offset）
    #       np.ndarray,  # Omni Hand qpos
    #     ]
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
    
    # 应用 max_objects 限制
    if max_objects is not None and max_objects > 0:
        object_items = list(object_id_to_indices.items())[:max_objects]
        cprint(f"[INFO] Limiting to {max_objects} object(s) (will process {len(object_items)} objects)", "yellow")
    else:
        object_items = list(object_id_to_indices.items())
    
    # 查找 omni hand 在 robots 列表中的索引（只需查找一次）
    omni_idx = None
    for robot_idx in range(1, len(robots)):  # 跳过 Shadow Hand (idx=0)
        if robots[robot_idx] == RobotName.omni:
            omni_idx = robot_idx
            break
    
    if omni_idx is None:
        cprint(f"[WARNING] Omni hand not found in robots list: {robots}", "yellow")
    
    # 批量处理每个 object_id
    objects_processed = 0
    continuous_idx = 0  # 连续索引计数器，从 0 开始
    for object_id, indices in tqdm(object_items, desc="Processing objects"):
        objects_processed += 1
        cprint(f"\n[INFO] Processing object {object_id} ({len(indices)} grasps) [{objects_processed}/{len(object_items)}]...", "cyan")
        
        for idx in tqdm(indices, desc=f"Object {object_id}", leave=False):
            try:
                # 加载数据样本
                data = dataset[idx]
                
                # 获取 Shadow Hand qpos（使用 pin_order 版本）
                shadow_qpos = data["grasp_qpos_pin_order"]  # (29,)
                grasp_qpos = data["grasp_qpos"]
                
                # 获取物体的base scale（obj_scale）
                scene_config = data["scene_config"]
                object_id_key = data["object_id"]
                obj_config = scene_config[object_id_key]
                obj_scale = np.array(obj_config["scale"])  # [sx, sy, sz]
                
                arr = scene_config[data["object_id"]]['pose']
                poses = [sapien.Pose(arr[:3].tolist(), arr[3:].tolist())]
                # 存储基本信息，使用连续索引作为键，并保存原始索引
                retargeted_poses_dict[continuous_idx] = {
                    "original_idx": idx,  # 保存原始数据集索引
                    "target_object_name": data["object_id"],
                    "grasp_type": data["grasp_type"],
                    "scale_name": data["scale_name"],
                    "grasp_idx": data["grasp_idx"],
                    "scene_scale": data["scene_scale"],  # 场景缩放因子（标量）
                    "obj_scale": obj_scale.copy(),  # 物体基础缩放 [sx, sy, sz]
                    "target_pose_world": poses,
                    "shadow_qpos": grasp_qpos.copy(),  # Shadow Hand 原始 qpos
                    "robot_pose": [],  # 存储 Omni Hand 的 retargeted qpos 列表
                }
                
                # 只对 omni hand 进行 retargeting
                if omni_idx is not None:
                    # 执行 retargeting（不包含 y_offset）
                    retargeted_qpos = retarget_shadow_to_robot_no_offset(
                        viewer, shadow_qpos, omni_idx
                    )
                    
                    if retargeted_qpos is not None:
                        retargeted_poses_dict[continuous_idx]["robot_pose"].append(retargeted_qpos.copy())
                        cprint(f"  [DEBUG] Retargeted to omni: qpos shape {retargeted_qpos.shape}", "white")
                    else:
                        cprint(f"  [WARNING] Failed to retarget to omni", "yellow")
                
                # 如果可视化，加载物体和手并渲染
                if visualize:
                    viewer.load_object(data)
                    viewer.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
                    # 设置 omni hand 的 qpos
                    if omni_idx is not None and len(retargeted_poses_dict[continuous_idx]["robot_pose"]) > 0:
                        omni_qpos = retargeted_poses_dict[continuous_idx]["robot_pose"][0]
                        viewer.robots[omni_idx].set_qpos(omni_qpos.astype(np.float32))
                    
                    # 创建关节位置标记（使用 retargeting 中实际使用的关节索引）
                    joint_positions_world = viewer._compute_shadow_joint_positions(shadow_qpos)
                    
                    # 获取 retargeting 中使用的关节索引（用于第一个机器人）
                    if len(robots) > 1:
                        retargeting = viewer.retargetings[1]  # 第一个 retargeting 目标机器人
                        if retargeting is not None:
                            indices = retargeting.optimizer.target_link_human_indices
                            retargeting_type = retargeting.optimizer.retargeting_type
                            
                            # 根据 retargeting 类型提取实际使用的关节索引
                            if retargeting_type == "POSITION" or retargeting_type == "FINGERTIP":
                                # 一维数组，直接使用
                                used_indices = np.unique(indices.flatten())
                            elif retargeting_type == "VECTOR" or retargeting_type == "DEXPILOT":
                                # 二维数组，提取所有 origin 和 task 索引
                                if len(indices.shape) == 2 and indices.shape[0] >= 2:
                                    origin_indices = indices[0, :]
                                    task_indices = indices[1, :]
                                    used_indices = np.unique(np.concatenate([origin_indices, task_indices]))
                                else:
                                    # 如果不是二维数组，使用所有索引
                                    used_indices = np.unique(indices.flatten())
                            else:
                                # 默认使用所有索引
                                used_indices = np.unique(indices.flatten())
                            
                            # 只显示使用的关节
                            joint_positions_to_show = joint_positions_world[used_indices]
                            cprint(f"  [DEBUG] Showing {len(used_indices)} joints used in retargeting (indices: {used_indices})", "white")
                        else:
                            # 如果没有 retargeting，显示所有关节
                            joint_positions_to_show = joint_positions_world
                            cprint(f"  [DEBUG] No retargeting found, showing all {len(joint_positions_world)} joints", "white")
                    else:
                        # 如果没有目标机器人，显示所有关节
                        joint_positions_to_show = joint_positions_world
                        cprint(f"  [DEBUG] No target robots, showing all {len(joint_positions_world)} joints", "white")
                    
                    viewer._create_joint_marker_spheres(joint_positions_to_show)
                    
                    viewer.scene.update_render()
                    
                    if not viewer.headless:
                        # 确保 viewer 不是 paused 状态
                        if hasattr(viewer, 'viewer') and viewer.viewer is not None:
                            viewer.viewer.paused = False
                            
                            # 持续渲染，但设置一个超时机制
                            # 渲染一段时间后自动继续（或者等待用户关闭窗口）
                            import time
                            render_start_time = time.time()
                            render_duration = 60.0  # 渲染 60 秒后自动继续
                            
                            cprint(f"  [INFO] Rendering data {idx} (object: {data['object_id']}, grasp: {data['grasp_type']})", "cyan")
                            cprint(f"  [INFO] Close window or wait {render_duration}s to continue...", "yellow")
                            
                            while not viewer.viewer.closed:
                                viewer.viewer.render()
                                # 如果超过指定时间，自动继续
                                if time.time() - render_start_time > render_duration:
                                    cprint(f"  [INFO] Timeout reached, continuing to next sample...", "yellow")
                                    break
                        else:
                            cprint(f"  [WARNING] Viewer not initialized, skipping visualization", "yellow")
                    else:
                        # Headless 模式：渲染几帧
                        frames_to_render = 50
                        for frame in range(frames_to_render):
                            viewer.scene.update_render()
                            viewer.camera.take_picture()
                    
                    # 成功处理后递增连续索引
                    continuous_idx += 1
                    
            except Exception as e:
                cprint(f"[ERROR] Failed to process data {idx}: {e}", "red")
                import traceback
                traceback.print_exc()
                # 处理失败时不递增索引，跳过该数据
                continue
        
        # 检查是否达到最大物体数量限制
        if max_objects is not None and objects_processed >= max_objects:
            cprint(f"\n[INFO] Reached max_objects limit ({max_objects}), stopping processing", "yellow")
            break
    
    # 保存结果
    if len(retargeted_poses_dict) > 0:
        save_name = input("\nEnter the name for the file (without extension): ")
        save_path = data_root / f"grasp_poses_retargeted_{save_name}.npy"
        
        # 转换为普通字典（移除 defaultdict）
        result_dict = dict(retargeted_poses_dict)

        # DEBUG
        # print("retargeted_poses_dict", retargeted_poses_dict[0]["shadow_qpos"])
        # print("retargeted_poses_dict", retargeted_poses_dict[0]["robot_pose"])

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
    max_objects: Optional[int] = None,
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
        max_objects: 最大处理的物体数量（None 表示处理所有物体，1 表示只处理第一个物体）
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
        max_objects=max_objects,
    )


if __name__ == "__main__":
    tyro.cli(main)

