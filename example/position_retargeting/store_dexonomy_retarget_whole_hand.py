"""
存储 Shadow Hand 到其他机器手的 retargeting 结果
模仿 store_hand_object.py 的逻辑，批量处理 Dexonomy 数据集
"""
from pathlib import Path
from typing import Any, Optional, List, Dict
from collections import defaultdict
import sys

import numpy as np
import tyro
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from termcolor import cprint
import torch
import torch.nn.functional as F
import trimesh

from dexonomy_dataset import DexonomyGraspDataset
from dexonomy_viewer import DexonomyGraspSAPIENViewer
from dex_retargeting.constants import RobotName, HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

import sapien
import pytorch_kinematics as pk
import transforms3d
import urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere

# 尝试导入 pytorch3d（用于 FPS 采样）
try:
    import pytorch3d.ops
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    cprint("[WARNING] pytorch3d not available, FPS sampling will be disabled", "yellow")

# For numpy version compatibility
np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
np.unicode = np.unicode_


def get_hand_surface_points_from_sapien(
    viewer: DexonomyGraspSAPIENViewer, 
    robot_idx: int = 0,
    num_samples: int = 2000,
    return_normals: bool = True
) -> tuple:
    """
    从 SAPIEN robot 中提取手部表面点（均匀采样）
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        robot_idx: 机器人索引（0 表示 Shadow Hand）
        num_samples: 采样点数量（默认 2000）
        return_normals: 是否返回法向量（默认 True）
    
    Returns:
        hand_surface_points: (N, 3) 手部表面点世界坐标
        hand_surface_normals: (N, 3) 手部表面法向量（单位向量），如果 return_normals=True
    """
    robot = viewer.robots[robot_idx]
    all_points = []
    all_normals = []
    
    # 遍历所有 link，提取 collision mesh 的顶点（作为表面点）
    for link in robot.get_links():
        # 获取 link 的世界坐标 pose
        link_pose = link.get_pose()
        
        # 尝试获取 collision shapes（更可靠）
        collision_shapes = link.get_collision_shapes()
        
        for shape in collision_shapes:
            # 直接从 shape 获取几何信息（不使用 .geometry）
            # 检查是否是 mesh 类型
            if hasattr(shape, 'vertices') and shape.vertices is not None:
                vertices = np.array(shape.vertices)
                
                if len(vertices) > 0:
                    # 获取 shape 的 local pose
                    shape_pose = shape.get_local_pose()
                    
                    # 转换到 link 坐标系
                    vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                    shape_transform = shape_pose.to_transformation_matrix()
                    vertices_link = (shape_transform @ vertices_homogeneous.T).T[:, :3]
                    
                    # 转换到世界坐标系
                    link_transform = link_pose.to_transformation_matrix()
                    vertices_homogeneous = np.hstack([vertices_link, np.ones((len(vertices_link), 1))])
                    vertices_world = (link_transform @ vertices_homogeneous.T).T[:, :3]
                    
                    all_points.append(vertices_world)
                    
                    # 计算法向量（简单估计：指向外部）
                    if return_normals:
                        # 对于 mesh，使用简单的法向量估计
                        center = vertices_world.mean(axis=0)
                        normals = vertices_world - center
                        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
                        all_normals.append(normals)
            # 如果是球体或其他简单几何体，采样表面点
            elif hasattr(shape, 'radius'):
                # 球体：在表面采样点
                radius = shape.radius
                # 使用 Fibonacci 球面采样
                n_samples = 100
                indices = np.arange(0, n_samples, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / n_samples)
                theta = np.pi * (1 + 5**0.5) * indices
                
                x = radius * np.cos(theta) * np.sin(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(phi)
                sphere_points = np.stack([x, y, z], axis=1)
                
                # 球体法向量就是归一化的位置向量
                sphere_normals = sphere_points / (np.linalg.norm(sphere_points, axis=1, keepdims=True) + 1e-8)
                
                # 转换到世界坐标
                shape_pose = shape.get_local_pose()
                vertices_homogeneous = np.hstack([sphere_points, np.ones((len(sphere_points), 1))])
                shape_transform = shape_pose.to_transformation_matrix()
                vertices_link = (shape_transform @ vertices_homogeneous.T).T[:, :3]
                
                link_transform = link_pose.to_transformation_matrix()
                vertices_homogeneous = np.hstack([vertices_link, np.ones((len(vertices_link), 1))])
                vertices_world = (link_transform @ vertices_homogeneous.T).T[:, :3]
                
                all_points.append(vertices_world)
                
                if return_normals:
                    # 转换法向量（只旋转，不平移）
                    normals_homogeneous = np.hstack([sphere_normals, np.zeros((len(sphere_normals), 1))])
                    normals_link = (shape_transform @ normals_homogeneous.T).T[:, :3]
                    normals_world = (link_transform[:3, :3] @ normals_link.T).T
                    normals_world = normals_world / (np.linalg.norm(normals_world, axis=1, keepdims=True) + 1e-8)
                    all_normals.append(normals_world)
                    
            elif hasattr(shape, 'half_lengths'):
                # 盒子：采样表面点
                half_lengths = shape.half_lengths
                # 在盒子表面采样
                n_per_face = 20
                box_points = []
                box_normals = []
                for i, length in enumerate(half_lengths):
                    # 生成两个面
                    if i == 0:  # x 方向
                        y = np.linspace(-half_lengths[1], half_lengths[1], n_per_face)
                        z = np.linspace(-half_lengths[2], half_lengths[2], n_per_face)
                        yy, zz = np.meshgrid(y, z)
                        box_points.append(np.stack([np.full_like(yy, length), yy, zz], axis=-1).reshape(-1, 3))
                        box_points.append(np.stack([np.full_like(yy, -length), yy, zz], axis=-1).reshape(-1, 3))
                        if return_normals:
                            box_normals.append(np.tile([1, 0, 0], (yy.size, 1)))
                            box_normals.append(np.tile([-1, 0, 0], (yy.size, 1)))
                    elif i == 1:  # y 方向
                        x = np.linspace(-half_lengths[0], half_lengths[0], n_per_face)
                        z = np.linspace(-half_lengths[2], half_lengths[2], n_per_face)
                        xx, zz = np.meshgrid(x, z)
                        box_points.append(np.stack([xx, np.full_like(xx, length), zz], axis=-1).reshape(-1, 3))
                        box_points.append(np.stack([xx, np.full_like(xx, -length), zz], axis=-1).reshape(-1, 3))
                        if return_normals:
                            box_normals.append(np.tile([0, 1, 0], (xx.size, 1)))
                            box_normals.append(np.tile([0, -1, 0], (xx.size, 1)))
                    else:  # z 方向
                        x = np.linspace(-half_lengths[0], half_lengths[0], n_per_face)
                        y = np.linspace(-half_lengths[1], half_lengths[1], n_per_face)
                        xx, yy = np.meshgrid(x, y)
                        box_points.append(np.stack([xx, yy, np.full_like(xx, length)], axis=-1).reshape(-1, 3))
                        box_points.append(np.stack([xx, yy, np.full_like(xx, -length)], axis=-1).reshape(-1, 3))
                        if return_normals:
                            box_normals.append(np.tile([0, 0, 1], (xx.size, 1)))
                            box_normals.append(np.tile([0, 0, -1], (xx.size, 1)))
                
                if len(box_points) > 0:
                    box_surface = np.vstack(box_points)
                    
                    # 转换到世界坐标
                    shape_pose = shape.get_local_pose()
                    vertices_homogeneous = np.hstack([box_surface, np.ones((len(box_surface), 1))])
                    shape_transform = shape_pose.to_transformation_matrix()
                    vertices_link = (shape_transform @ vertices_homogeneous.T).T[:, :3]
                    
                    link_transform = link_pose.to_transformation_matrix()
                    vertices_homogeneous = np.hstack([vertices_link, np.ones((len(vertices_link), 1))])
                    vertices_world = (link_transform @ vertices_homogeneous.T).T[:, :3]
                    
                    all_points.append(vertices_world)
                    
                    if return_normals:
                        box_normals_array = np.vstack(box_normals)
                        normals_homogeneous = np.hstack([box_normals_array, np.zeros((len(box_normals_array), 1))])
                        normals_link = (shape_transform @ normals_homogeneous.T).T[:, :3]
                        normals_world = (link_transform[:3, :3] @ normals_link.T).T
                        normals_world = normals_world / (np.linalg.norm(normals_world, axis=1, keepdims=True) + 1e-8)
                        all_normals.append(normals_world)
    
    if len(all_points) == 0:
        cprint("[WARNING] No surface points extracted from hand!", "yellow")
        if return_normals:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.zeros((0, 3))
    
    # 合并所有点
    hand_surface_points = np.vstack(all_points)
    
    if return_normals:
        hand_surface_normals = np.vstack(all_normals) if len(all_normals) > 0 else None
        if hand_surface_normals is None or len(hand_surface_normals) != len(hand_surface_points):
            # 如果法向量缺失或不匹配，使用简单估计
            center = hand_surface_points.mean(axis=0)
            hand_surface_normals = hand_surface_points - center
            hand_surface_normals = hand_surface_normals / (np.linalg.norm(hand_surface_normals, axis=1, keepdims=True) + 1e-8)
    
    # 均匀采样到指定数量
    if len(hand_surface_points) != num_samples:
        if len(hand_surface_points) > num_samples:
            indices = np.random.choice(len(hand_surface_points), num_samples, replace=False)
        else:
            indices = np.random.choice(len(hand_surface_points), num_samples, replace=True)
        
        hand_surface_points = hand_surface_points[indices]
        if return_normals:
            hand_surface_normals = hand_surface_normals[indices]
    
    if return_normals:
        return hand_surface_points, hand_surface_normals
    return hand_surface_points


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    从 6D 旋转表示计算旋转矩阵（不导入 HandModel，直接实现）
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    def normalize_vector(v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, v.new([1e-8]))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v/v_mag
        return v

    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        return out

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw)
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def get_hand_surface_points_from_handmodel(
    grasp_qpos: np.ndarray,
    robot_name: str = 'shadowhand',
    urdf_path: Optional[str] = None,
    mesh_path: Optional[str] = None,
    num_samples: int = 2000,
    device: str = 'cuda',
    return_normals: bool = True
) -> tuple:
    """
    直接实现 HandModel 的表面点提取逻辑（不导入 HandModel 类）
    与 HandModel.get_surface_points_new() 方法逻辑一致
    
    采样策略：
    - 第一个有效 link（手掌）：采样 num_samples - (有效link数-1) * per_link_samples 个点
    - 其他 link（手指等）：各采样 per_link_samples 个点（shadow hand: 64，其他: 128）
    - 最后微调总点数到精确的 num_samples
    
    Args:
        grasp_qpos: (29,) Shadow Hand qpos，包含 [x, y, z, qw, qx, qy, qz, joint_angles...]
        robot_name: 机器人名称
        urdf_path: URDF 文件路径
        mesh_path: Mesh 文件路径
        num_samples: 总采样点数（默认 2000）
        device: 计算设备
        return_normals: 是否返回法向量（默认 True）
    
    Returns:
        hand_surface_points: (N, 3) 手部表面点世界坐标
        hand_surface_normals: (N, 3) 手部表面法向量（单位向量），如果 return_normals=True
    """
    try:
        # 默认路径设置
        if urdf_path is None:
            gendexgrasp_root = Path(__file__).absolute().parent.parent.parent.parent / "GenDexGrasp"
            urdf_path = str(gendexgrasp_root / f"data/urdf/{robot_name}.urdf")
        
        if mesh_path is None:
            gendexgrasp_root = Path(__file__).absolute().parent.parent.parent.parent / "GenDexGrasp"
            mesh_path = str(gendexgrasp_root / f"data/urdf/{robot_name}_meshes")
        
        # 检查文件是否存在
        if not Path(urdf_path).exists():
            cprint(f"[WARNING] URDF file not found: {urdf_path}", "yellow")
            return np.zeros((0, 3))
        
        if not Path(mesh_path).exists():
            cprint(f"[WARNING] Mesh path not found: {mesh_path}", "yellow")
            return np.zeros((0, 3))
        
        # 设置设备
        torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
        hand_scale = 1.0
        
        # 1. 加载 URDF 和创建运动学链
        # 以二进制模式读取，避免 XML 编码声明问题
        with open(urdf_path, 'rb') as f:
            urdf_data = f.read()
        robot = pk.build_chain_from_urdf(urdf_data).to(dtype=torch.float, device=torch_device)
        visual = URDF.from_xml_string(urdf_data)
        
        # 2. 转换 qpos 格式
        # grasp_qpos: [x, y, z, qw, qx, qy, qz, joint_angles...]
        translation = grasp_qpos[:3]  # (3,)
        quat = grasp_qpos[3:7]  # (qw, qx, qy, qz)
        joint_angles = grasp_qpos[7:]  # 关节角度
        
        # 四元数 → 旋转矩阵（直接使用，无需转 6D）
        rotation_scipy = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy 格式: [qx, qy, qz, qw]
        rot_matrix = rotation_scipy.as_matrix()  # (3, 3) numpy array
        
        # 3. 更新运动学
        global_translation = torch.from_numpy(translation).unsqueeze(0).float().to(torch_device)  # (1, 3)
        global_rotation = torch.from_numpy(rot_matrix).unsqueeze(0).float().to(torch_device)  # (1, 3, 3)
        q_joints = torch.from_numpy(joint_angles).unsqueeze(0).float().to(torch_device)  # (1, 22)
        
        current_status = robot.forward_kinematics(q_joints)
        
        # 4. 提取每个 link 的表面点和法向量
        # 先统计有效 link 数量（有 visual 的）
        valid_links = [link for link in visual.links if len(link.visuals) > 0]
        num_valid_links = len(valid_links)
        
        # 确定每个普通 link 的采样数（根据目标总点数动态调整）
        if robot_name == 'shadowhand':
            if num_samples <= 2000:
                per_link_samples = 64
            elif num_samples <= 5000:
                per_link_samples = 200
            else:
                # 对于更大的采样数，按比例缩放（5000点对应200，向上线性外推）
                per_link_samples = int(200 * (num_samples / 5000))
        else:
            # 其他机器人：保持原有逻辑
            per_link_samples = 128
        
        # 第一个 link（手掌）采样更多点
        if num_valid_links > 0:
            first_link_samples = num_samples - (num_valid_links - 1) * per_link_samples
            first_link_samples = max(first_link_samples, per_link_samples)  # 至少采样 per_link_samples 个点
        else:
            first_link_samples = per_link_samples
        
        surface_points_list = []
        surface_normals_list = []
        
        valid_link_idx = 0  # 有效 link 的索引
        for i_link, link in enumerate(visual.links):
            # 跳过没有 visual 的 link
            if len(link.visuals) == 0:
                continue
            
            # 加载 mesh
            if type(link.visuals[0].geometry) == Mesh:
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                else:
                    filename = link.visuals[0].geometry.filename
                mesh_file = Path(mesh_path) / filename
                if not mesh_file.exists():
                    continue
                mesh = trimesh.load(str(mesh_file), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = trimesh.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, 
                    height=link.visuals[0].geometry.length
                )
            elif type(link.visuals[0].geometry) == Box:
                mesh = trimesh.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = trimesh.primitives.Sphere(radius=link.visuals[0].geometry.radius)
            else:
                continue
            # 获取 scale
            try:
                scale = np.array(link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            
            # 获取 origin (rotation 和 translation)
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation_link = np.reshape(link.visuals[0].origin.xyz, [1, 3])
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation_link = np.array([[0, 0, 0]])
            
            # 采样表面点（第一个有效 link 采样更多点）
            if valid_link_idx == 0:
                # 第一个 link（通常是手掌）
                sample_count = first_link_samples
            else:
                # 其他 link
                sample_count = per_link_samples
            
            pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=sample_count)
            valid_link_idx += 1  # 递增有效 link 计数器
            
            # 获取法向量
            if return_normals:
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            
            # 应用 scale
            pts *= scale
            
            # Shadow Hand 特殊处理：坐标轴转换
            # if robot_name == 'shadowhand':
            #     pts = pts[:, [0, 2, 1]]
            #     pts[:, 1] *= -1
            #     if return_normals:
            #         pts_normal = pts_normal[:, [0, 2, 1]]
            #         pts_normal[:, 1] *= -1
            
            # 转换到 link 坐标系
            pts = np.matmul(rotation, pts.T).T + translation_link
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)  # 齐次坐标
            
            if return_normals:
                # 法向量只旋转，不平移
                pts_normal = np.matmul(rotation, pts_normal.T).T
                pts_normal = np.concatenate([pts_normal, np.zeros([len(pts_normal), 1])], axis=-1)
            
            # 转换为 torch tensor
            pts_tensor = torch.from_numpy(pts).to(torch_device).float().unsqueeze(0)  # (1, N, 4)
            if return_normals:
                pts_normal_tensor = torch.from_numpy(pts_normal).to(torch_device).float().unsqueeze(0)  # (1, N, 4)
            
            # 转换到世界坐标系
            if link.name in current_status:
                trans_matrix = current_status[link.name].get_matrix()  # (1, 4, 4)
                # 应用 link 变换
                pts_transformed = torch.matmul(trans_matrix, pts_tensor.transpose(1, 2)).transpose(1, 2)[..., :3]  # (1, N, 3)
                # 应用全局旋转和平移
                pts_world = torch.matmul(global_rotation, pts_transformed.transpose(1, 2)).transpose(1, 2) + global_translation.unsqueeze(1)
                # 应用 hand_scale
                pts_world = pts_world * hand_scale
                surface_points_list.append(pts_world.squeeze(0))  # (N, 3)
                
                if return_normals:
                    # 法向量：只应用旋转
                    pts_normal_transformed = torch.matmul(trans_matrix, pts_normal_tensor.transpose(1, 2)).transpose(1, 2)[..., :3]
                    pts_normal_world = torch.matmul(global_rotation, pts_normal_transformed.transpose(1, 2)).transpose(1, 2)
                    # 归一化
                    pts_normal_world = pts_normal_world / (torch.norm(pts_normal_world, dim=2, keepdim=True) + 1e-8)
                    surface_normals_list.append(pts_normal_world.squeeze(0))  # (N, 3)
        
        # 5. 合并所有 link 的表面点
        if len(surface_points_list) == 0:
            cprint(f"[WARNING] No surface points extracted from any link", "yellow")
            if return_normals:
                return np.zeros((0, 3)), np.zeros((0, 3))
            return np.zeros((0, 3))
        
        surface_points = torch.cat(surface_points_list, dim=0)  # (total_N, 3)
        if return_normals:
            surface_normals = torch.cat(surface_normals_list, dim=0)  # (total_N, 3)
        
        # 6. 确保精确的点数（通常已经接近目标，只需微调）
        # 采样策略：第一个 link 采样 first_link_samples 个，其他 link 各采样 per_link_samples 个
        current_num = surface_points.shape[0]
        cprint(f"[INFO] Extracted {current_num} surface points from {num_valid_links} links (target: {num_samples})", "cyan")
        
        if current_num != num_samples:
            if current_num > num_samples:
                # 降采样：随机选择
                indices = torch.randperm(current_num, device=torch_device)[:num_samples]
                surface_points = surface_points[indices]
                if return_normals:
                    surface_normals = surface_normals[indices]
                cprint(f"[INFO] Adjusted from {current_num} to {num_samples} points (downsampled)", "white")
            else:
                # 上采样：重复采样
                indices = torch.randint(0, current_num, (num_samples,), device=torch_device)
                surface_points = surface_points[indices]
                if return_normals:
                    surface_normals = surface_normals[indices]
                cprint(f"[INFO] Adjusted from {current_num} to {num_samples} points (upsampled)", "white")
        
        # 转换为 numpy
        surface_points_np = surface_points.cpu().numpy()  # (N, 3)
        if return_normals:
            surface_normals_np = surface_normals.cpu().numpy()  # (N, 3)
        
        cprint(f"[SUCCESS] Final: {surface_points_np.shape[0]} surface points", "green")
        
        if return_normals:
            return surface_points_np, surface_normals_np
        return surface_points_np
        
    except Exception as e:
        cprint(f"[ERROR] Failed to extract surface points: {e}", "red")
        import traceback
        traceback.print_exc()
        if return_normals:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.zeros((0, 3))


def get_object_point_cloud_and_normals(
    viewer: DexonomyGraspSAPIENViewer, 
    object_idx: int = 0, 
    num_samples: int = 2048, 
    mesh_path: Optional[str] = None, 
    obj_scale: Optional[np.ndarray] = None,
    use_fps: bool = False,
    device: str = 'cuda'
) -> tuple:
    """
    从 SAPIEN 物体中提取点云和法线
    
    Args:
        viewer: DexonomyGraspSAPIENViewer 实例
        object_idx: 物体索引
        num_samples: 采样点数量
        mesh_path: mesh 文件路径（如果提供，直接从文件加载）
        obj_scale: 物体缩放因子（如果提供，应用到 mesh）
        use_fps: 是否使用 FPS (Farthest Point Sampling) 进行均匀采样（需要 pytorch3d）
        device: 计算设备（'cuda' 或 'cpu'）
    
    Returns:
        point_cloud: (num_samples, 3) 物体点云世界坐标
        normal_cloud: (num_samples, 3) 物体法线世界坐标
    """
    if object_idx >= len(viewer.objects):
        cprint(f"[WARNING] Object index {object_idx} out of range!", "yellow")
        return np.zeros((num_samples, 3)), np.zeros((num_samples, 3))
    
    obj = viewer.objects[object_idx]
    obj_pose = obj.get_pose()
    
    # 方法 1: 如果提供了 mesh_path，直接从文件加载（最可靠）
    if mesh_path is not None and Path(mesh_path).exists():
        try:
            # 使用 trimesh 加载 mesh
            mesh = trimesh.load(str(mesh_path), force='mesh')
            
            # 如果是 Scene，获取第一个几何体
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]
            
            # 应用缩放
            if obj_scale is not None:
                # trimesh apply_scale 支持标量或数组
                if isinstance(obj_scale, np.ndarray) and len(obj_scale) == 3:
                    # 数组形式 [sx, sy, sz]
                    mesh.apply_scale(obj_scale)
                else:
                    # 标量形式
                    mesh.apply_scale(float(obj_scale))
            
            # 获取顶点和法线
            vertices = np.array(mesh.vertices)
            
            # 计算顶点法线（如果不存在）
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = np.array(mesh.vertex_normals)
            else:
                # 使用 trimesh 计算法线
                mesh.fix_normals()
                normals = np.array(mesh.vertex_normals)
            
            # 转换到世界坐标系
            obj_transform = obj_pose.to_transformation_matrix()
            vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
            vertices_world = (obj_transform @ vertices_homogeneous.T).T[:, :3]
            
            # 转换法线（只旋转，不平移）
            obj_rotation = obj_transform[:3, :3]
            normals_world = (obj_rotation @ normals.T).T
            
            all_vertices = vertices_world
            all_normals = normals_world
            
        except Exception as e:
            cprint(f"[WARNING] Failed to load mesh from {mesh_path}: {e}", "yellow")
            # Fallback 到方法 2
            all_vertices = []
            all_normals = []
    else:
        # 方法 2: 尝试从 Actor 的 visual bodies 提取
        all_vertices = []
        all_normals = []
        
        # Actor 对象可能没有 get_collision_shapes，尝试 visual bodies
        try:
            # 尝试获取 visual bodies
            if hasattr(obj, 'get_visual_bodies'):
                visual_bodies = obj.get_visual_bodies()
                for visual_body in visual_bodies:
                    # 尝试获取 render shapes
                    if hasattr(visual_body, 'get_render_shapes'):
                        render_shapes = visual_body.get_render_shapes()
                        for render_shape in render_shapes:
                            if hasattr(render_shape, 'mesh') and render_shape.mesh is not None:
                                mesh = render_shape.mesh
                                if hasattr(mesh, 'vertices') and mesh.vertices is not None:
                                    vertices = np.array(mesh.vertices)
                                    if len(vertices) > 0:
                                        # 获取 visual body 的 local pose
                                        visual_pose = visual_body.local_pose
                                        
                                        # 转换到世界坐标系
                                        vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                                        visual_transform = visual_pose.to_transformation_matrix()
                                        vertices_obj = (visual_transform @ vertices_homogeneous.T).T[:, :3]
                                        
                                        obj_transform = obj_pose.to_transformation_matrix()
                                        vertices_homogeneous = np.hstack([vertices_obj, np.ones((len(vertices_obj), 1))])
                                        vertices_world = (obj_transform @ vertices_homogeneous.T).T[:, :3]
                                        
                                        all_vertices.append(vertices_world)
                                        
                                        # 获取法线
                                        if hasattr(mesh, 'normals') and mesh.normals is not None:
                                            normals = np.array(mesh.normals)
                                            visual_rotation = visual_transform[:3, :3]
                                            normals_obj = (visual_rotation @ normals.T).T
                                            obj_rotation = obj_transform[:3, :3]
                                            normals_world = (obj_rotation @ normals_obj.T).T
                                            all_normals.append(normals_world)
            elif hasattr(obj, 'get_visual_shapes'):
                # 尝试 get_visual_shapes
                visual_shapes = obj.get_visual_shapes()
                for shape in visual_shapes:
                    if hasattr(shape, 'vertices') and shape.vertices is not None:
                        vertices = np.array(shape.vertices)
                        if len(vertices) > 0:
                            shape_pose = shape.get_local_pose()
                            vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
                            shape_transform = shape_pose.to_transformation_matrix()
                            vertices_obj = (shape_transform @ vertices_homogeneous.T).T[:, :3]
                            
                            obj_transform = obj_pose.to_transformation_matrix()
                            vertices_homogeneous = np.hstack([vertices_obj, np.ones((len(vertices_obj), 1))])
                            vertices_world = (obj_transform @ vertices_homogeneous.T).T[:, :3]
                            all_vertices.append(vertices_world)
        except Exception as e:
            cprint(f"[WARNING] Failed to extract from Actor visual bodies: {e}", "yellow")
    
    # 如果仍然没有顶点，返回空数组
    if len(all_vertices) == 0:
        cprint("[WARNING] No vertices extracted from object! Try providing mesh_path.", "yellow")
        return np.zeros((num_samples, 3)), np.zeros((num_samples, 3))
    
    # 合并所有顶点
    if isinstance(all_vertices, list):
        all_vertices = np.vstack(all_vertices)
    if isinstance(all_normals, list) and len(all_normals) > 0:
        all_normals = np.vstack(all_normals)
    elif len(all_normals) == 0:
        # 如果没有法线，使用简单估计
        center = all_vertices.mean(axis=0)
        normals = all_vertices - center
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        all_normals = normals
    
    # 确保顶点和法线数量一致
    if len(all_normals) != len(all_vertices):
        if len(all_normals) < len(all_vertices):
            # 用简单估计填充
            center = all_vertices.mean(axis=0)
            missing_normals = all_vertices[len(all_normals):] - center
            missing_normals = missing_normals / (np.linalg.norm(missing_normals, axis=1, keepdims=True) + 1e-8)
            all_normals = np.vstack([all_normals, missing_normals]) if len(all_normals) > 0 else missing_normals
        else:
            all_normals = all_normals[:len(all_vertices)]
    
    # 采样到指定数量
    if len(all_vertices) != num_samples:
        if len(all_vertices) > num_samples:
            # 降采样
            if use_fps and HAS_PYTORCH3D:
                # 使用 FPS (Farthest Point Sampling) 进行均匀采样
                cprint(f"  [INFO] Using FPS sampling to downsample from {len(all_vertices)} to {num_samples} points", "cyan")
                torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
                
                # 转换为 torch tensor
                vertices_tensor = torch.from_numpy(all_vertices).float().unsqueeze(0).to(torch_device)  # (1, N, 3)
                
                # 使用 FPS 采样
                sampled_points, sampled_indices = pytorch3d.ops.sample_farthest_points(
                    vertices_tensor, 
                    K=num_samples,
                    random_start_point=True
                )
                
                # 转换为 numpy
                sampled_indices_np = sampled_indices.squeeze(0).cpu().numpy()  # (num_samples,)
                point_cloud = all_vertices[sampled_indices_np]
                normal_cloud = all_normals[sampled_indices_np]
                
                cprint(f"  [INFO] FPS sampling completed: {point_cloud.shape}", "green")
            else:
                # 随机采样（原有方法）
                if use_fps:
                    cprint(f"  [WARNING] pytorch3d not available, falling back to random sampling", "yellow")
                indices = np.random.choice(len(all_vertices), num_samples, replace=False)
                point_cloud = all_vertices[indices]
                normal_cloud = all_normals[indices]
        else:
            # 上采样：重复采样
            indices = np.random.choice(len(all_vertices), num_samples, replace=True)
            point_cloud = all_vertices[indices]
            normal_cloud = all_normals[indices]
    else:
        point_cloud = all_vertices
        normal_cloud = all_normals
    
    # 归一化法线
    normal_norms = np.linalg.norm(normal_cloud, axis=1, keepdims=True)
    normal_norms = np.where(normal_norms == 0, 1, normal_norms)
    normal_cloud = normal_cloud / normal_norms
    
    return point_cloud, normal_cloud


def compute_contact_map_on_object(
    hand_surface_points: np.ndarray,
    hand_surface_normals: np.ndarray,
    object_point_cloud: np.ndarray,
    object_normal_cloud: np.ndarray,
    contact_threshold: float = 0.02,
    use_torch: bool = True
) -> np.ndarray:
    """
    计算物体点云上的 contact map（从手到物体）
    这是 GenDexGrasp CMapAdam 需要的格式
    
    Args:
        hand_surface_points: (N_hand, 3) 手部表面点
        hand_surface_normals: (N_hand, 3) 手部表面法向量
        object_point_cloud: (N_obj, 3) 物体点云
        object_normal_cloud: (N_obj, 3) 物体法线
        contact_threshold: 接触阈值
        use_torch: 是否使用 PyTorch（更快）
    
    Returns:
        contact_value_on_object: (N_obj,) 每个物体点的接触值（0-1）
    """
    if use_torch:
        # 转换为 torch tensor
        hand_points = torch.from_numpy(hand_surface_points).float()  # (N_hand, 3)
        hand_normals = torch.from_numpy(hand_surface_normals).float()  # (N_hand, 3)
        obj_points = torch.from_numpy(object_point_cloud).float()    # (N_obj, 3)
        obj_normals = torch.from_numpy(object_normal_cloud).float()  # (N_obj, 3)
        
        npts_hand = hand_points.shape[0]
        npts_object = obj_points.shape[0]
        
        # 计算物体点到手表面点的对齐距离
        # obj_points: (1, 1, N_obj, 3) -> (1, N_hand, N_obj, 3)
        # hand_points: (1, N_hand, 1, 3) -> (1, N_hand, N_obj, 3)
        obj_expanded = obj_points.unsqueeze(0).unsqueeze(1)  # (1, 1, N_obj, 3)
        hand_expanded = hand_points.unsqueeze(0).unsqueeze(2)  # (1, N_hand, 1, 3)
        hand_normals_expanded = hand_normals.unsqueeze(0).unsqueeze(2)  # (1, N_hand, 1, 3)
        
        # 计算距离
        hand_obj_dist = (obj_expanded - hand_expanded).norm(dim=3)  # (1, N_hand, N_obj)
        
        # 计算对齐分数（使用手的法线方向）
        hand_obj_align = ((obj_expanded - hand_expanded) * hand_normals_expanded).sum(dim=3)  # (1, N_hand, N_obj)
        hand_obj_align = hand_obj_align / (hand_obj_dist + 1e-5)
        
        # 计算对齐距离
        hand_obj_align_dist = hand_obj_dist * torch.exp(2 * (1 - torch.abs(hand_obj_align)))
        
        # 对每个物体点，取到最近手表面点的距离
        contact_dist = torch.sqrt(hand_obj_align_dist.min(dim=1)[0])  # (1, N_obj)
        
        # 计算接触值（sigmoid 函数）
        contact_value = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)  # (1, N_obj)
        
        return contact_value.squeeze(0).numpy()  # (N_obj,)
    else:
        # NumPy 实现（较慢，用于调试）
        npts_hand = hand_surface_points.shape[0]
        npts_object = object_point_cloud.shape[0]
        
        contact_values = np.zeros(npts_object)
        
        for i in range(npts_object):
            obj_point = object_point_cloud[i]  # (3,)
            
            # 计算到所有手部点的距离
            diffs = hand_surface_points - obj_point[None, :]  # (N_hand, 3)
            dists = np.linalg.norm(diffs, axis=1)  # (N_hand,)
            
            # 计算对齐分数（使用手的法线）
            aligns = np.sum(-diffs * hand_surface_normals, axis=1) / (dists + 1e-5)  # (N_hand,)
            
            # 计算对齐距离
            align_dists = dists * np.exp(2 * (1 - np.abs(aligns)))  # (N_hand,)
            
            # 取最小距离
            min_dist = np.sqrt(np.min(align_dists))
            
            # 计算接触值
            contact_values[i] = 1 - 2 * (1 / (1 + np.exp(-10 * min_dist)) - 0.5)
        
        return contact_values


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
    use_handmodel_method: bool = True,
    hand_surface_sample_num: int = 2000,
    use_fps_for_object: bool = False,
    visualize_hand_points: bool = False,
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
        use_handmodel_method: 是否使用 HandModel 方法提取手部表面点（True）或使用 SAPIEN 方法（False）
        hand_surface_sample_num: 手部表面采样点数（默认 2000）
        use_fps_for_object: 是否使用 FPS (Farthest Point Sampling) 对物体点云进行均匀采样（需要 pytorch3d）
        visualize_hand_points: 是否在 SAPIEN 中可视化手部表面点（需要 visualize=True）
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
                
                # 获取 Shadow Hand qpos（三种 order）
                # shadow_qpos: pinocchio order - 用于 retargeting 和 pytorch_kinematics (HandModel)
                # grasp_qpos: sapien order - 用于 SAPIEN viewer
                shadow_qpos = data["grasp_qpos_pin_order"]  # (29,) pinocchio order
                grasp_qpos = data["grasp_qpos"]  # (29,) sapien order
                bodex_qpos = data["grasp_qpos_bodex_order"]  # (29,) bodex original order
                pk_qpos = data["grasp_pos_pk_order"]  # (29,) pk order
                # 获取物体的base scale（obj_scale）
                scene_config = data["scene_config"]
                object_id_key = data["object_id"]
                obj_config = scene_config[object_id_key]
                obj_scale = np.array(obj_config["scale"])  # [sx, sy, sz]
                
                arr = scene_config[data["object_id"]]['pose']
                poses = [sapien.Pose(arr[:3].tolist(), arr[3:].tolist())]
                
                # 计算 contact map（在加载物体后）
                contact_map = None
                hand_surface_points = None
                hand_surface_normals = None
                object_point_cloud = np.array([])
                object_normal_cloud = np.array([])
                try:
                    # 加载物体以便提取点云
                    viewer.load_object(data)
                    # SAPIEN viewer 使用 sapien order 的 qpos
                    viewer.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
                    
                    # 根据参数选择方法提取手部表面点和法向量
                    if use_handmodel_method:
                        # 使用 HandModel 方法
                        cprint(f"  [INFO] Using HandModel method to extract hand surface points...", "cyan")
                        try:
                            # 注意：pytorch_kinematics 需要使用 pinocchio order 的 qpos
                            hand_surface_points, hand_surface_normals, _ = get_hand_surface_points_from_handmodel(
                                grasp_qpos=pk_qpos,  # 使用 pinocchio order (29维)
                                robot_name='shadowhand',
                                urdf_path="/home/guizhewei/guizhewei/retarget/dex-retargeting/assets/robots/hands/shadow_hand_no_wrist/shadow_hand_right.urdf",
                                mesh_path="/home/guizhewei/guizhewei/retarget/dex-retargeting/assets/robots/hands/shadow_hand_no_wrist/meshes/visual",
                                num_samples=hand_surface_sample_num,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                return_normals=True
                            )
                        except Exception as e:
                            cprint(f"  [WARNING] HandModel method failed: {e}, falling back to SAPIEN method", "yellow")
                            hand_surface_points, hand_surface_normals, _ = get_hand_surface_points_from_sapien(
                                viewer, 
                                robot_idx=0,
                                num_samples=hand_surface_sample_num,
                                return_normals=True
                            )
                    else:
                        # 使用 SAPIEN 方法
                        cprint(f"  [INFO] Using SAPIEN method to extract hand surface points...", "cyan")
                        hand_surface_points, hand_surface_normals, _ = get_hand_surface_points_from_sapien(
                            viewer, 
                            robot_idx=0,
                            num_samples=hand_surface_sample_num,
                            return_normals=True
                        )
                    
                    # 检查是否成功提取
                    if hand_surface_points is None or len(hand_surface_points) == 0:
                        cprint(f"  [WARNING] Failed to extract hand surface points", "yellow")
                        hand_surface_points = np.zeros((0, 3))
                        hand_surface_normals = np.zeros((0, 3))
                    
                    # 提取物体点云和法线
                    # 注意：缩放应该与 viewer.load_object 中的一致
                    # obj_scale 是 [sx, sy, sz] 数组，scene_scale 是标量
                    actual_scale = obj_scale * data["scene_scale"]  # [sx*scene_scale, sy*scene_scale, sz*scene_scale]
                    cprint(f"  [DEBUG] Object scale: base={obj_scale}, scene_scale={data['scene_scale']}, actual={actual_scale}", "white")
                    object_point_cloud, object_normal_cloud = get_object_point_cloud_and_normals(
                        viewer, 
                        object_idx=0, 
                        num_samples=2048,
                        mesh_path=data.get("object_mesh_path"),
                        obj_scale=actual_scale,  # 应用缩放（与 viewer.load_object 一致）
                        use_fps=use_fps_for_object,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    
                    # 计算物体点云上的 contact map（GenDexGrasp 格式）
                    contact_map_object = np.array([])
                    
                    if len(hand_surface_points) > 0 and len(object_point_cloud) > 0:
                        # 计算物体点云的接触值（GenDexGrasp 格式）
                        contact_map_object = compute_contact_map_on_object(
                            hand_surface_points,
                            hand_surface_normals,
                            object_point_cloud,
                            object_normal_cloud,
                            contact_threshold=0.02,
                            use_torch=True
                        )
                        cprint(f"  [INFO] Computed contact map on object: shape {contact_map_object.shape}, mean={contact_map_object.mean():.4f}, max={contact_map_object.max():.4f}", "green")
                        cprint(f"  [INFO] Hand surface points: {hand_surface_points.shape}, normals: {hand_surface_normals.shape}", "white")
                    else:
                        cprint(f"  [WARNING] Cannot compute contact map: hand_points={len(hand_surface_points)}, obj_points={len(object_point_cloud)}", "yellow")
                except Exception as e:
                    cprint(f"  [ERROR] Failed to compute contact map for idx {idx}: {e}", "red")
                    import traceback
                    traceback.print_exc()
                    contact_map_object = np.array([])
                
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
                    # Contact map 数据（物体点云上的接触值，GenDexGrasp 格式）
                    "contact_map_object": contact_map_object if len(contact_map_object) > 0 else np.array([]),  # 物体点云的接触值 (M,)
                    # 物体数据
                    "object_point_cloud": object_point_cloud if len(object_point_cloud) > 0 else np.array([]),  # 物体点云 (M, 3)
                    "object_normal_cloud": object_normal_cloud if len(object_normal_cloud) > 0 else np.array([]),  # 物体法向量 (M, 3)
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
                        cprint(f"  [WARNING] Failed to retarget to omni for idx {idx}", "yellow")
                        # 即使 retargeting 失败，也保留数据（robot_pose 为空列表）
                else:
                    cprint(f"  [WARNING] Omni hand not found, skipping retargeting for idx {idx}", "yellow")
                    # 即使没有 omni hand，也保留数据（robot_pose 为空列表）
                
                # 如果可视化，渲染场景（物体已经在计算 contact map 时加载）
                if visualize:
                    # 物体已经加载，只需设置手的 qpos（SAPIEN viewer 使用 sapien order）
                    viewer.set_shadow_qpos(grasp_qpos, robot_idx=0, y_offset=0.0)
                    
                    # 设置 omni hand 的 qpos（如果不可视化手部表面点）
                    if not visualize_hand_points:
                        # 正常显示 omni hand
                        if omni_idx is not None and len(retargeted_poses_dict[continuous_idx]["robot_pose"]) > 0:
                            omni_qpos = retargeted_poses_dict[continuous_idx]["robot_pose"][0]
                            viewer.robots[omni_idx].set_qpos(omni_qpos.astype(np.float32))
                    else:
                        # 可视化手部表面点时，隐藏 omni hand（移到远处）
                        if omni_idx is not None:
                            dummy_qpos = np.zeros(viewer.robots[omni_idx].dof)
                            dummy_qpos[:3] = [100.0, 100.0, 100.0]  # 移到远处
                            viewer.robots[omni_idx].set_qpos(dummy_qpos.astype(np.float32))
                            cprint(f"  [INFO] Hiding omni hand (moved to [100, 100, 100]) to show hand surface points", "cyan")
                    
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
                    
                    # 可视化手部表面点（如果启用）
                    if visualize_hand_points and len(hand_surface_points) > 0:
                        cprint(f"  [INFO] Visualizing {len(hand_surface_points)} hand surface points...", "cyan")
                        
                        # 清除现有的 hand point markers
                        if not hasattr(viewer, 'hand_point_markers'):
                            viewer.hand_point_markers = []
                        else:
                            for marker in viewer.hand_point_markers:
                                viewer.scene.remove_actor(marker)
                            viewer.hand_point_markers.clear()
                        
                        # 如果点数太多，采样一部分（避免渲染太慢）
                        max_vis_points = 20000
                        if len(hand_surface_points) > max_vis_points:
                            indices = np.random.choice(len(hand_surface_points), max_vis_points, replace=False)
                            vis_hand_points = hand_surface_points[indices]
                            cprint(f"  [INFO] Sampled {max_vis_points} points from {len(hand_surface_points)} total hand points", "white")
                        else:
                            vis_hand_points = hand_surface_points
                        
                        # 创建球体 markers（橙红色，与物体点云区分）
                        sphere_radius = 0.003  # 3mm
                        hand_color = np.array([1.0, 0.4, 0.2, 0.8])  # 橙红色 RGBA
                        
                        for i, pos in enumerate(vis_hand_points):
                            # 创建球体 material
                            sphere_material = sapien.render.RenderMaterial()
                            sphere_material.set_base_color(hand_color)
                            sphere_material.set_roughness(0.3)
                            sphere_material.set_metallic(0.1)
                            sphere_material.set_specular(0.8)
                            
                            # 创建球体
                            builder = viewer.scene.create_actor_builder()
                            builder.add_sphere_visual(radius=sphere_radius, material=sphere_material)
                            marker = builder.build_static(name=f"hand_point_marker_{i}")
                            marker.set_pose(sapien.Pose(pos))
                            viewer.hand_point_markers.append(marker)
                        
                        cprint(f"  [INFO] Created {len(viewer.hand_point_markers)} hand surface point markers", "green")
                    
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
                
                # 成功处理后递增连续索引（无论是否可视化都要递增）
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
    cprint(f"\n[INFO] Processing completed. Total entries in dict: {len(retargeted_poses_dict)}", "green")
    if len(retargeted_poses_dict) > 0:
        # 转换为普通字典（移除 defaultdict）
        result_dict = dict(retargeted_poses_dict)
        cprint(f"[INFO] Converted to regular dict. Entries: {len(result_dict)}", "green")
        
        # 检查第一个条目
        if 0 in result_dict:
            cprint(f"[DEBUG] First entry keys: {list(result_dict[0].keys())}", "white")
            cprint(f"[DEBUG] First entry robot_pose length: {len(result_dict[0].get('robot_pose', []))}", "white")
        
        save_name = input("\nEnter the name for the file (without extension): ")
        save_path = data_root / f"grasp_poses_retargeted_{save_name}.npy"

        np.save(save_path, result_dict)
        cprint(f"\n[INFO] Saved {len(result_dict)} retargeted poses to {save_path}", "green")
        cprint(f"[INFO] Robot names: {[str(r) for r in robots]}", "cyan")
        cprint(f"[INFO] Retargeting type: {retargeting_type}", "cyan")
        if two_optimizers:
            cprint(f"[INFO] Second optimizer type: {second_optimizer_type}", "cyan")
    else:
        cprint("[WARNING] No data to save! Dictionary is empty.", "yellow")
        cprint("[WARNING] This might indicate all data processing failed or no data was processed.", "yellow")


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
    use_handmodel_method: bool = True,
    hand_surface_sample_num: int = 5000,
    use_fps_for_object: bool = True,
    visualize_hand_points: bool = False,
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
        use_handmodel_method: 是否使用 HandModel 方法提取手部表面点（True）或使用 SAPIEN 方法（False）
        hand_surface_sample_num: 手部表面采样点数（默认 2000）
        use_fps_for_object: 是否使用 FPS (Farthest Point Sampling) 对物体点云进行均匀采样（需要 pytorch3d）
        visualize_hand_points: 是否在 SAPIEN 中可视化手部表面点（需要 visualize=True，橙红色球体）
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
        use_handmodel_method=use_handmodel_method,
        hand_surface_sample_num=hand_surface_sample_num,
        use_fps_for_object=use_fps_for_object,
        visualize_hand_points=visualize_hand_points,
    )


if __name__ == "__main__":
    tyro.cli(main)

