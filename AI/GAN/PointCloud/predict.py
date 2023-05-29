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

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("club chair that is average size and regular height and regular width and long"))).astype(np.float64)).reshape(1,512, 1).repeat(32,1,1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh = generate_mesh(pcd)

    o3d.io.write_triangle_mesh("model.obj", mesh)
