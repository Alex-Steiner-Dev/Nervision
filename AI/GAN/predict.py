import torch
from model import Generator
from text_to_vec import *
from mesh_generation import *

import numpy as np
import pyvista as pv

Generator = Generator().cuda()

model_path = "../TrainedModels/model.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

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

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    mesh = o3d.io.read_triangle_mesh("dataset/40f1be4ede6113a2e03aea0698586c31.obj")
    simplified_mesh = mesh.simplify_quadric_decimation(2048)
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

    print(np.array(simplified_mesh.vertices)[0], vertices[0])

    point_cloud = np.array(np.array(simplified_mesh.vertices), dtype=np.float32)
    
    mesh = create_mesh(vertices, simplified_mesh.triangles)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    o3d.visualization.draw_geometries([pcd, mesh])