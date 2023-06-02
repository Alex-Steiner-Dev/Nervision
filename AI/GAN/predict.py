import torch
from model import *
from text_to_vec import *
import open3d as o3d

import numpy as np

Generator = Generator().cuda()
Autoencoder = Autoencoder().cuda()

vertices_path = "../TrainedModels/vertices.pt" 
checkpoint = torch.load(vertices_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

autoencoder_path = "../TrainedModels/autoencoder.pt" 
checkpoint = torch.load(autoencoder_path)
Autoencoder.load_state_dict(checkpoint['autoencoder'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("cocktail table that is tall and square and average size"))).astype(np.float64)).reshape(1,512, 1).repeat(1, 1, 1).cuda().float()

def create_mesh(vertices, faces):
    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh

def predict():
    with torch.no_grad():
        sample = Generator(z).cpu()
        vertices = sample.numpy()[0]

        vertices = np.array(vertices, dtype=np.float32)
        vertices = Autoencoder(torch.from_numpy(vertices).float().to('cuda')).cpu().detach().numpy()
        
        mesh = o3d.io.read_triangle_mesh("dataset/40f1be4ede6113a2e03aea0698586c31.obj")
        simplified_mesh = mesh.simplify_quadric_decimation(2048)

        return vertices, np.array(simplified_mesh.triangles)
    
v, f = predict()
mesh = create_mesh(v, f)

o3d.visualization.draw_geometries([mesh])