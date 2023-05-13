import torch.utils.data as data
from scipy.interpolate import interp1d
import trimesh
import json
from text_to_vec import *
import numpy as np
import torch
import logging

from sklearn.cluster import MiniBatchKMeans

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

            vertices, _ = trimesh.sample.sample_surface(mesh, count=50000)
            point_cloud = np.array(vertices, dtype=np.float32)

            point_cloud = self.reduce_dimension(point_cloud, num_points=2048)

            self.points.append(point_cloud)
            self.text_embeddings.append(label)

        f.close()

    def reduce_dimension(self, point_cloud, num_points):
        kmeans = MiniBatchKMeans(n_clusters=num_points, random_state=0)
        kmeans.fit(point_cloud)

        cluster_centers = kmeans.cluster_centers_

        return cluster_centers
        
    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.points[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.points)