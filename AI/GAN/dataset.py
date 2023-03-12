import os
from torch.utils.data import Dataset
import trimesh
import numpy as np
import torch

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


class LoadDataset (Dataset):
    def __init__(self, root="data/"):
        self.root = root
        self.listdir = os.listdir(self.root)

        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) 

    def __getitem__(self, index):
        volume = mesh_to_voxel(self.root + self.listdir[index])

        return volume

    def __len__(self):
        return len(self.listdir)