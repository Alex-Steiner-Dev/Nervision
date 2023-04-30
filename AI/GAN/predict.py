import torch
from mesh_generation import generate_mesh
from model import Generator
import open3d as o3d
from text_to_vec import *
import time

import pyvista as pv
import numpy as np

Generator = Generator().cuda()

model_path = "50.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("a bowl"))) + np.random.normal(0, 0.01, 512).astype(np.float64)).reshape(1,1,512).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    points = sample.numpy().reshape(2048,3)

    mesh = pv.PolyData(points).delaunay_3d().extract_geometry().smooth(n_iter=100)
    texture = pv.read_texture('texture.jpg')

    mesh.textures['texture'] = texture
    mesh.texture_map_to_plane(inplace=True)

    p = pv.Plotter()
    p.add_mesh(mesh)

    p.export_gltf("model.gltf")
    p.show()
