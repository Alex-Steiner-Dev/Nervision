import torch.utils.data as data
import open3d as o3d
import trimesh
import json
import os
from text_to_vec import *
import numpy as np
import logging
import objaverse

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadDataset(data.Dataset):
    def __init__(self, data_dir):
        self.objects = []
        self.labels = []

        f = open("captions.json")
        self.data = json.load(f)

        self.uids = []

        for i in self.data:
            self.uids.append(i)

        for i, glb_file in enumerate(os.listdir(data_dir)):
            glb_path = os.path.join(data_dir, glb_file)
   
            label = text_to_vec(process_text(self.data[self.uids[i]]['caption']))

            self.objects.append(glb_path)
            self.labels.append(label)
    
    def __getitem__(self, idx):
        obj_path = self.objects[idx]
        label = self.labels[idx]

        mesh = objaverse.load_objects([self.uids[idx]])
        mesh = o3d.io.read_triangle_mesh(obj_path)

        o3d.io.write_triangle_mesh(obj_path.replace(".glb", ".obj"), mesh)

        mesh = trimesh.load(obj_path.replace(".glb", ".obj"), force="mesh")

        points = mesh.sample(2048)
        points = np.array(points, dtype=np.float32)

        return points, label
    
    def __len__(self):
        return len(self.objects)
