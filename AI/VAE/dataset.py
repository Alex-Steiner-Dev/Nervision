import trimesh
import glob
import numpy as np

DATA_DIR = "../Data/VolumetricData/*"

def point_cloud_to_voxel_to_point_cloud(mesh_path):
    points = trimesh.load(mesh_path).sample(2048).T

    voxels = np.zeros((2048, 2048, 2048))
    voxels[0,:,:] = points[0]
    voxels[1,:,:] = points[1]
    voxels[2,:,:] = points[2]

    point_cloud = np.zeros((points.shape[1], 3))
    for i in range(3):
        flattened_slice = voxels[i, :, :].flatten()
        point_cloud[:, i] = flattened_slice[:points.shape[1]]

    return point_cloud, voxels

def parse_dataset():
    print("Loading dataset...")
    
    point_cloud, voxels = point_cloud_to_voxel_to_point_cloud("chair_0001.off")

    print("Done!")

    return voxels