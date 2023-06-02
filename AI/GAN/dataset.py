import torch.utils.data as data
import trimesh
import json
import open3d as o3d
from text_to_vec import *
import numpy as np
import torch
import logging

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadVertices(data.Dataset):
    def __init__(self):
        self.points = []
        self.text_embeddings = []

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            if i == 0:
                obj_path = "dataset/" + itObject['id'] + ".obj"

                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))
                
                mesh = o3d.io.read_triangle_mesh(obj_path)
                simplified_mesh = mesh.simplify_quadric_decimation(2048)

                if len(simplified_mesh.vertices) > 2048:
                    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

                vertices = np.array(simplified_mesh.vertices)
    
                expanded_array = np.zeros((2048, 3))
                expanded_array[:vertices.shape[0], :] = vertices
                
                point_cloud = np.array(expanded_array, dtype=np.float32)

                self.points.append(point_cloud)
                self.text_embeddings.append(label)

        f.close()
  
    def __getitem__(self, idx):
        point_cloud = torch.tensor(self.points[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return point_cloud, text_embedding
    
    def __len__(self):
        return len(self.points)
    
class LoadFaces(data.Dataset):
    def __init__(self):
        self.faces = []
        self.text_embeddings = []

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            if i == 0:
                obj_path = "dataset/" + itObject['id'] + ".obj"

                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))
                
                mesh = o3d.io.read_triangle_mesh(obj_path)
                simplified_mesh = mesh.simplify_quadric_decimation(2048)

                faces = np.array(simplified_mesh.triangles)

                print(faces.shape)
    
                expanded_array = np.zeros((2048, 3))
                expanded_array[:faces.shape[0], :] = faces
                
                faces = np.array(expanded_array, dtype=np.float32)

                self.faces.append(faces)
                self.text_embeddings.append(label)

        f.close()
  
    def __getitem__(self, idx):
        faces = torch.tensor(self.faces[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return faces, text_embedding
    
    def __len__(self):
        return len(self.faces)