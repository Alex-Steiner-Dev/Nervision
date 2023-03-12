import trimesh
import glob
import numpy as np
import pyvista as pv

DATA_DIR = "../Data/VolumetricData/*"

def mesh_to_voxel(mesh_path):
    points = trimesh.load(mesh_path).sample(256).T

    voxels = np.zeros((256, 256, 256))
    voxels[0,:,:] = points[0]
    voxels[1,:,:] = points[1]
    voxels[2,:,:] = points[2]

    return voxels

def voxel_to_point_cloud(voxels):
    reshaped_arr = np.zeros((voxels.shape[1], 3))
    for i in range(3):
        flattened_slice = voxels[i, :, :].flatten()
        reshaped_arr[:, i] = flattened_slice[:voxels.shape[1]]

    return reshaped_arr

def plot_point_cloud(points):
    point_cloud = pv.PolyData(points)
    point_cloud.plot()

def parse_dataset():
    print("Loading dataset...")
    voxels = []

    for i in range(1):
        voxel = mesh_to_voxel("chair_0001.off")
        voxels.append(voxel)

    print("Done!")

    return voxels