import torch
from model import Generator
from text_to_vec import *
from mesh_generation import *

import numpy as np
import pyvista as pv

Generator = Generator().cuda()

model_path = "../../TrainedModels/model.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

def create_mesh(vertices, faces):
    # Convert vertices and faces to NumPy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    
    # Set the vertices and faces of the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Update the mesh to compute the normals and other properties
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("cocktail table that is tall and square and average size"))).astype(np.float64)).reshape(1,512, 1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    obj_path = "../dataset/40f1be4ede6113a2e03aea0698586c31.obj"
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(2048)
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01)
    count = 0.01
    while len(simplified_mesh.vertices) > 2048:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01 / count)
        count = count - 0.01

    mesh = create_mesh(vertices, simplified_mesh.triangles)

    o3d.visualization.draw_geometries([mesh])
    

    pv.PolyData(vertices).plot()