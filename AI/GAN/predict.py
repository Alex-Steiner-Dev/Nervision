import torch
from mesh_generation import generate_mesh
from model import Generator
import open3d as o3d
from text_to_vec import *
import time

import pyvista as pv
import numpy as np

Generator = Generator().cuda()

model_path = "../TrainedModels/model.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt(""))) + np.random.normal(0, 0.01, 512).astype(np.float64)).reshape(1,1,512).cuda().float()

with torch.no_grad():
    start = time.time()

    sample = Generator(z).cpu()

    points = sample.numpy().reshape(2048,3)

    end = time.time()
    print("Time taken:" +  str(end - start))

    mesh = generate_mesh(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_triangle_mesh("mesh.obj", mesh)


mesh = pv.read('mesh.obj')
texture = pv.read_texture('texture.jpeg')

mesh.textures['texture'] = texture
mesh.texture_map_to_plane(inplace=True)

mesh.plot()
mesh.save('mesh_uv.stl')