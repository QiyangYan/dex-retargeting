import numpy as np

# import a npy file from the given path
path = "/home/ubuntu/Documents/DexYCB/grasp_poses.npy"
grasp_pose = np.load(path, allow_pickle=True).item()
sample = grasp_pose["0"]

print(sample['robot_pose'][:3])
print(sample['target_pose_world'])