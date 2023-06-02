import torch
import model
from text_to_vec import *
import open3d as o3d

import numpy as np

Generator = model.Generator().cuda()
Autoencoder = model.Autoencoder().cuda()

vertices_path = "../TrainedModels/vertices.pt" 
checkpoint = torch.load(vertices_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

autoencoder_path = "../TrainedModels/autoencoder.pt" 
checkpoint_ae = torch.load(autoencoder_path)
Autoencoder.load_state_dict(checkpoint_ae['autoencoder'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("ping pong table that is regular height and average size"))).astype(np.float64)).reshape(1,512, 1).repeat(1, 1, 1).cuda().float()

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
        points = sample.numpy()[0]

    vertices = Autoencoder(torch.from_numpy(points).to('cuda')).cpu().detach().numpy()
    vertices = np.array(vertices, dtype=np.float32)
        
    mesh = o3d.io.read_triangle_mesh("dataset/97deac79537426ac9255fc5df0de0bff.obj")
    simplified_mesh = mesh.simplify_quadric_decimation(2048)

    if len(simplified_mesh.vertices) > 2048:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

    faces = np.array(simplified_mesh.triangles)
 
    mesh = create_mesh(vertices, faces)

    o3d.visualization.draw_geometries([mesh])

predict()