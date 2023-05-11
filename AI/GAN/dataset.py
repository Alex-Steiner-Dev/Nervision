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
        self.paths = []
        self.text_embeddings = []

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            obj_path = "dataset/" + itObject['mid'] + ".obj"

            if itObject['desc'].split('.')[0].find(".") != -1:
                label = text_to_vec(process_text(itObject['desc']))
            else:
                label = text_to_vec(process_text(itObject['desc'].split('.')[0]))

            self.paths.append(obj_path)
            self.text_embeddings.append(label)

        f.close()
        
    def __getitem__(self, idx):
        mesh = trimesh.load(self.paths[idx], force="mesh")
        vertices, _ = trimesh.sample.sample_surface(mesh, count=40960)
        point_cloud = np.array(vertices, dtype=np.float32)
        point_cloud = torch.from_numpy(point_cloud)
        
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.paths)