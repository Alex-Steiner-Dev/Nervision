import torch.utils.data as data
import pyvista as pv
import json
import open3d as o3d
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
                obj_path = "../dataset/" + itObject['id'] + ".obj"

                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))
                
                mesh = o3d.io.read_triangle_mesh(obj_path)

                simplified_mesh = mesh.simplify_quadric_decimation(2048)
                simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01)
                count = 0.01

                while len(simplified_mesh.vertices) > 2048:
                    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01 / count)
                    count = count - 0.01

                x = np.array(simplified_mesh.vertices)
                point_cloud = np.vstack([x, np.zeros((2048 - x.shape[0], 3))])
                point_cloud = np.array(point_cloud, dtype=np.float32)

                self.points.append(point_cloud)
                self.text_embeddings.append(label)

        f.close()
  
    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.points[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.points)