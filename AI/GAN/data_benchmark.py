import torch.utils.data as data
import trimesh
import glob
import torch
import numpy as np

class LoadDataset(data.Dataset):
    def __init__(self, root, npoints=2048):
        self.npoints = npoints
        self.root = root
        self.meshes = glob.glob(self.root + "*.obj")
        self.labels = glob.glob(self.root + "*.txt")

    def __getitem__(self, index):
        mesh = self.meshes[index]
        label = self.labels[index]

        points = trimesh.Trimesh(mesh, process=False).sample(self.npoints)

        return points, label

    def __len__(self):
        return len(self.meshes)