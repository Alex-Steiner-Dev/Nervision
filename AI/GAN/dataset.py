import torch.utils.data as data
import trimesh
import json
from text_to_vec import *
import numpy as np
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadDataset(data.Dataset):
    def __init__(self, data_dir):
        self.points = []
        self.labels = []

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            obj_path = "dataset/" + itObject['mid'] + ".obj"

            if itObject['desc'].split('.')[0].find(".") != -1:
                label = text_to_vec(process_text(itObject['desc']))
            else:
                label = text_to_vec(process_text(itObject['desc'].split('.')[0]))

            mesh = trimesh.load(obj_path, force="mesh")

            point = mesh.sample(2048)
            point = np.array(point, dtype=np.float32)

            self.points.append(point)
            self.labels.append(label)

        f.close()
        
    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]

        return points, label
    
    def __len__(self):
        return len(self.points)
