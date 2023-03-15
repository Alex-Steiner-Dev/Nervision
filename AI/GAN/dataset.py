import os
import trimesh
import numpy as np

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

