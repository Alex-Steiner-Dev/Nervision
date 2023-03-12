import trimesh
import numpy as np
import pyvista as pv

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