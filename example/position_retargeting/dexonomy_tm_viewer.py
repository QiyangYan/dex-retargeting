import trimesh
import numpy as np
import os
from glob import glob
import logging
import multiprocessing
import mujoco
import transforms3d.quaternions as tq

from dexonomy_dataset import DexonomyGraspDataset

class RobotKinematics:
    def __init__(self, xml_path):
        spec = mujoco.MjSpec.from_file(xml_path)
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        self.mesh_geom_info = {}
        for i in range(self.mj_model.ngeom):
            geom = self.mj_model.geom(i)
            mesh_id = geom.dataid
            if mesh_id != -1:
                mjm = self.mj_model.mesh(mesh_id)
                vert = self.mj_model.mesh_vert[
                    mjm.vertadr[0] : mjm.vertadr[0] + mjm.vertnum[0]
                ]
                face = self.mj_model.mesh_face[
                    mjm.faceadr[0] : mjm.faceadr[0] + mjm.facenum[0]
                ]
                body_name = self.mj_model.body(geom.bodyid).name
                mesh_name = mjm.name
                self.mesh_geom_info[f"{body_name}_{mesh_name}"] = {
                    "vert": vert,
                    "face": face,
                    "geom_id": i,
                }

        return

    def forward_kinematics(self, q):
        self.mj_data.qpos = q
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        return

    def get_init_meshes(self):
        init_mesh_lst = []
        mesh_name_lst = []
        for k, v in self.mesh_geom_info.items():
            mesh_name_lst.append(k)
            init_mesh_lst.append(trimesh.Trimesh(vertices=v["vert"], faces=v["face"]))
        return mesh_name_lst, init_mesh_lst

    def get_poses(self, root_pose):
        geom_poses = np.zeros((len(self.mesh_geom_info), 7))
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        for i, v in enumerate(self.mesh_geom_info.values()):
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_poses[i, :3] = root_rot @ geom_trans + root_trans
            geom_poses[i, 3:] = tq.mat2quat(root_rot @ geom_rot)
        return geom_poses

    def get_posed_meshes(self, root_pose):
        root_rot = tq.quat2mat(root_pose[3:])
        root_trans = root_pose[:3]
        full_tm = []
        for k, v in self.mesh_geom_info.items():
            geom_rot = self.mj_data.geom_xmat[v["geom_id"]].reshape(3, 3)
            geom_trans = self.mj_data.geom_xpos[v["geom_id"]]
            posed_vert = (v["vert"] @ geom_rot.T + geom_trans) @ root_rot.T + root_trans
            posed_tm = trimesh.Trimesh(vertices=posed_vert, faces=v["face"])
            full_tm.append(posed_tm)
        full_tm = trimesh.util.concatenate(full_tm)
        return full_tm

def _single_visd(params):

    from pathlib import Path

    hand_fk = RobotKinematics("/home/guizhewei/guizhewei/retarget/dex-retargeting/assets/robots/hands/shadow_hand_noforearm_xml/right_hand.xml")
    data_root = Path("/home/guizhewei/guizhewei/Dexonomy_dataset")

    dataset = DexonomyGraspDataset(
        data_root=data_root,
        grasp_type="5_Light_Tool",  # 修正：使用 grasp_type（单数）
        split="train",
    )

    grasp_data = dataset[50]


    # Visualize hand
    all_qpos = grasp_data["grasp_qpos"]
    all_qpos = all_qpos[None] if len(all_qpos.shape) == 1 else all_qpos

    for i in range(all_qpos.shape[0]):
        hand_pose = all_qpos[i, :7]
        hand_qpos = all_qpos[i, 7:]


        hand_fk.forward_kinematics(hand_qpos)
        visual_mesh = hand_fk.get_posed_meshes(hand_pose)

        # visual_mesh.export(f"hand_tmp.obj")

    from termcolor import cprint
    cprint(grasp_data["scene_config"][grasp_data["object_id"]]["pose"], "red") 

    # Visualize object
    obj_path = os.path.join(grasp_data["object_mesh_path"])
    obj_tm = trimesh.load(obj_path, force="mesh")
    obj_tm.vertices *= (grasp_data["scene_config"][grasp_data["object_id"]]["scale"] * grasp_data["scene_scale"])
    # rotation_matrix = trimesh.transformations.quaternion_matrix(
    #     grasp_data["obj_pose"][3:]
    # )
    # rotation_matrix[:3, 3] = grasp_data["obj_pose"][:3]
    # obj_tm.apply_transform(rotation_matrix)
    # obj_tm.export(f"{out_path}_obj.obj")

    tm_scene = trimesh.Scene([obj_tm] +[visual_mesh])
    tm_scene.show()

    # logging.info(f"Save to {os.path.dirname(out_path)}")

    return

if __name__ == "__main__":
    _single_visd(None)