import torch.utils.data as data
import json
import open3d as o3d
from text_to_vec import *
import numpy as np
import torch
import logging
import model

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

class LoadVertices(data.Dataset):
    def __init__(self):
        self.points = []
        self.text_embeddings = []

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            obj_path = "dataset/" + itObject['id'] + ".obj"
     
            mesh = o3d.io.read_triangle_mesh(obj_path)
            simplified_mesh = mesh.simplify_quadric_decimation(4096)

            if len(simplified_mesh.vertices) > 4096:
                simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

            if len(simplified_mesh.vertices) < 4096:
                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))

                vertices = np.array(simplified_mesh.vertices)
        
                expanded_array = np.zeros((4096, 3))
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
                obj_path = "dataset/" + itObject['id'] + ".obj"

                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(itObject['desc']))
                else:
                    label = text_to_vec(process_text(itObject['desc'].split('.')[0]))
                
                mesh = o3d.io.read_triangle_mesh(obj_path)
                simplified_mesh = mesh.simplify_quadric_decimation(4096)

                faces = np.array(simplified_mesh.triangles)
    
                expanded_array = np.zeros((4096, 3))
                expanded_array[:faces.shape[0], :] = faces
                
                faces = (np.array(expanded_array, dtype=np.float32) + np.random.normal(loc=0, scale=0.4, size=(4096,3))) * 10
                faces = np.array(faces, dtype=np.float32)

                self.faces.append(faces)
                self.text_embeddings.append(label)

        f.close()
  
    def __getitem__(self, idx):
        faces = torch.tensor(self.faces[idx])
        text_embedding = torch.tensor(self.text_embeddings[idx])

        return faces, text_embedding
    
    def __len__(self):
        return len(self.faces)
    
class LoadAutoEncoder(data.Dataset):
    def __init__(self):
        self.target = []
        self.generated = []

        Generator = model.Generator().cuda()

        vertices_path = "../TrainedModels/vertices.pt" 
        checkpoint = torch.load(vertices_path)
        Generator.load_state_dict(checkpoint['G_state_dict'])

        f = open("captions.json")
        self.data = json.load(f)

        for i, itObject in enumerate(self.data):
            obj_path = "dataset/" + itObject['id'] + ".obj"
     
            mesh = o3d.io.read_triangle_mesh(obj_path)
            simplified_mesh = mesh.simplify_quadric_decimation(4096)

            if len(simplified_mesh.vertices) > 4096:
                simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

            if len(simplified_mesh.vertices) < 4096:
                if itObject['desc'].split('.')[0].find(".") != -1:
                    label = text_to_vec(process_text(correct_prompt(itObject['desc'])))
                else:
                    label = text_to_vec(process_text(correct_prompt(itObject['desc'].split('.')[0])))

                z = torch.from_numpy(label.astype(np.float64)).reshape(1,512, 1).repeat(13, 1, 1).cuda().float()
                
                with torch.no_grad():
                    sample = Generator(z).cpu()
                    points = sample.numpy()[0]

                generated = np.array(points)
                vertices = np.array(simplified_mesh.vertices)

                expanded_array = np.zeros((4096, 3))
                expanded_array[:vertices.shape[0], :] = vertices
                    
                vertices = np.array(expanded_array, dtype=np.float32)

                self.target.append(vertices)
                self.generated.append(generated)

        f.close()
  
    def __getitem__(self, idx):
        target = torch.tensor(self.target[idx])
        generated = torch.tensor(self.generated[idx])

        return generated, target
    
    def __len__(self):
        return len(self.target)