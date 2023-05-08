import torch.utils.data as data
from scipy.interpolate import interp1d
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

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            obj_path = "dataset/" + itObject['mid'] + ".obj"

            if itObject['desc'].split('.')[0].find(".") != -1:
                label = text_to_vec(process_text(itObject['desc']))
            else:
                label = text_to_vec(process_text(itObject['desc'].split('.')[0]))

            mesh = trimesh.load(obj_path, force="mesh")

            point_cloud = mesh.sample(100000)
            point_cloud = np.array(point_cloud, dtype=np.float32)

            point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 4096, replace=False), :]

            self.points.append(point_cloud)
            self.text_embeddings.append(label)

        f.close()
        
    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.points[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.points)