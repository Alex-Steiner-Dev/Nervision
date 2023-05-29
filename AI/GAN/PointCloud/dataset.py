import torch.utils.data as data
import trimesh
import json
from text_to_vec import *
import numpy as np
import torch
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadDataset(data.Dataset):
    def __init__(self, data_dir):
        self.points = []
        self.text_embeddings = []

        f = open("../captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            if i % 32 == 0:
                obj_path = "../dataset/" + itObject['mid'] + ".obj"

                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))

                mesh = trimesh.load(obj_path, force="mesh")

                target_reduction = 1.0 - (2048 / len(mesh.points))
                simplified_mesh = mesh.decimate(target_reduction=target_reduction, inplace=True)
                point_cloud = np.array(simplified_mesh.points, dtype=np.float32)

                self.points.append(point_cloud)
                self.text_embeddings.append(label)

        f.close()
        
    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.points[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.points)