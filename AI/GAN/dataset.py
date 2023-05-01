import torch.utils.data as data
import trimesh
import open3d as o3d
import os
from text_to_vec import *
import numpy as np
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadDataset(data.Dataset):
    def __init__(self, data_dir):
        self.objects = []
        self.labels = []

        for glb_file in os.listdir(data_dir):
            if glb_file.endswith('.glb'):

                glb_path = os.path.join(data_dir, glb_file)
   
                label = text_to_vec(process_text(f.read()))

                self.objects.append(glb_path)
                self.labels.append(label)
    
    def __getitem__(self, idx):
        obj_path = self.objects[idx]
        label = self.labels[idx]

        mesh = trimesh.load(obj_path, force="mesh")

        points = mesh.sample(2048)
        points = np.array(points, dtype=np.float32)

        return points, label
    
    def __len__(self):
        return len(self.objects)
