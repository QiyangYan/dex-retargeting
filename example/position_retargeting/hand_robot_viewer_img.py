import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import sapien
from torch import obj
from hand_viewer import HandDatasetSAPIENViewer
from pytransform3d import rotations
from tqdm import trange

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from termcolor import cprint
from dataset import YCB_CLASSES


class RobotHandDatasetSAPIENViewer_IMG(HandDatasetSAPIENViewer):
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless=False,
        use_ray_tracing=False,
        retargeting_type=RetargetingType.position,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        

        # Load optimizer and filter
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, retargeting_type, hand_type
            )
            ''' Add another optimzer '''
            vector_config_path = get_default_config_path(
                    robot_name, RetargetingType.dexpilot, hand_type
                )
            override = dict(add_dummy_free_joint=True)
            cprint(f"Config path: {vector_config_path}", "yellow")
            config = RetargetingConfig.load_from_file(vector_config_path, override=override)
            self.vector_retargeting = config.build()

            # Add 6-DoF dummy joint at the root of each robot to make them move freely in the space
            override = dict(add_dummy_free_joint=True)
            cprint(f"Config path: {config_path}", "yellow")
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            robot_file_name = Path(config.urdf_path).stem
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_name(urdf_path.stem + "_glb" + urdf_path.suffix)
            cprint(f"urdf path: {urdf_path}", "yellow")
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)

            cprint(f"Robot path: {temp_path}", "yellow")
            robot = loader.load(temp_path)
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        super().load_object_hand(data)
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        """
        hand_pose: (num_frame, 1, 51) holding the pose of the hand. 
        pose_m[:, 0:48] stores the MANO pose coefficients in PCA representation, and pose_m[0, 48:51] stores the translation.
        
        joint: [num_frame, 21, 3] holding the 3D joint position of the hand in the camera coordinates. 21 links
        If the image does not have a visible hand or the annotation does not exist, joint_3d will be all -1.
        
        qpos: [num_frame, 18]: 7 wrist pose + 11 joint
        
        """
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)
        else:
            local_pose = self.camera.get_local_pose()
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))
            self.camera.set_local_pose(local_pose)

        hand_pose = data["hand_pose"]
        object_pose = data["object_pose"]
        num_frame = hand_pose.shape[0]
        num_copy = len(self.robots) + 1
        num_ycb_objects = len(data["ycb_ids"])
        pose_offsets = []

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]
            robot_names = "_".join(robot_names)
            video_path = (
                Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            )
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        # Warm start
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )
        vertex, joint = self._compute_hand_geometry(hand_pose_start)
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )

        # Loop rendering
        step_per_frame = int(60 / fps)
        # for i in trange(num_frame-1, num_frame):
            
        i = num_frame - 1
        
        object_pose_frame = object_pose[i] # current frame
        hand_pose_frame = hand_pose[i]
        vertex, joint = self._compute_hand_geometry(hand_pose_frame)

        obj_pose_camera_frame = object_pose_frame.copy()
        obj_pose_world_frame = []

        # Update poses for YCB objects
        for k in range(num_ycb_objects):
            pos_quat = object_pose_frame[k]

            # Quaternion convention: xyzw -> wxyz
            pose = self.camera_pose * sapien.Pose(
                pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
            )

            # Set original object's pose — no lifting
            self.objects[k].set_pose(pose)

            # Only apply offset and z-lift to copies
            for copy_ind in range(1, num_copy):  # skip 0
                offset_pose = pose_offsets[copy_ind] * pose

                # Modify z
                # pos = offset_pose.p.copy()
                # pos[:3] = 0
                # offset_pose = sapien.Pose(p=pos, q=offset_pose.q)

                self.objects[k + copy_ind * num_ycb_objects].set_pose(offset_pose)

            obj_pose_world_frame.append(offset_pose)

        # Update pose for human hand
        self._update_hand(vertex)

        # Update poses for robot hands
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices # predefined mapping indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint[indices, :] # target link's 3D position, (5, 3)
            else: # vector retargeting
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = (
                    joint[task_indices, :] - joint[origin_indices, :]
                ) 
            qpos = retargeting.retarget(ref_value)[retarget2sapien] # (18, )
            # qpos[1] += 0.8  # Set the root position to zero

            '''Vector retargeting for finger'''
            retargeting_type = self.vector_retargeting.optimizer.retargeting_type
            indices = self.vector_retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = (
                joint[task_indices, :] - joint[origin_indices, :]
            ) 
            qpos_vector = self.vector_retargeting.retarget(ref_value)[retarget2sapien]
            qpos[7:] = qpos_vector[7:] # only replace the finger joints

            ''' Set joint '''
            robot.set_qpos(qpos) 

        self.scene.update_render()
        if self.headless:
            self.camera.take_picture()
            rgb = self.camera.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])
        else:
            for _ in range(step_per_frame):
                self.viewer.render()

        if not self.headless:
            self.viewer.paused = True
            self.viewer.render()
        else:
            writer.release()
        
        # the idx of moved obj
        target_object_idx = np.where(~np.all(object_pose[0] == object_pose[-1], axis=1))[0][0]
        target_object_name = YCB_CLASSES[data["ycb_ids"][target_object_idx]]

        cprint(f"Robot {robot.name} qpos: {qpos} target: {target_object_name} target object pose: {obj_pose_world_frame[target_object_idx]}", "yellow")

        # store as a dict
        pose = {
            # target object
            "target_object_name": target_object_name,
            "target_object_idx": data["ycb_ids"][target_object_idx],
            
            # object pose and camera pose
            "object_pose_camera_frame": object_pose_frame,
            "object_pose_world_frame": obj_pose_world_frame, # object pose for robot
            "camera_pose": self.camera_pose,

            # robot pose in object frame

            
            # robot info
            "robot_names": self.robot_names,
            "robot_qpos": qpos, # robot pose : 7 pose + 11 joint 
            "hand_type": self.hand_type,
            "mano_pose": hand_pose_frame[0],

            # retargeting method
            "retargeting_type": retargeting_type,
        }
