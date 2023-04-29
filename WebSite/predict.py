import torch
import sys

sys.path.append("../AI/GAN/")

from mesh_generation import generate_mesh
from text_to_vec import *

import numpy as np
import pyvista as pv

from model import Generator
import open3d as o3d

Generator = Generator().cuda()

model_path = "../AI/TrainedModels/model.pt" 

checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

def generate(text):
    z = torch.from_numpy(text_to_vec(process_text(correct_prompt(text))) + np.random.normal(0, 0.01, 512).astype(np.float64)).reshape(1,1,512).cuda().float()

    with torch.no_grad():
        sample = Generator(z).cpu()

        points = sample.numpy().reshape(2048,3)

        mesh = generate_mesh(points)

        o3d.io.write_triangle_mesh("static/generations/generation_" + sys.argv[2] + ".obj", mesh)

        mesh = pv.read("static/generations/generation_" + sys.argv[2] + ".obj")
        texture = pv.read_texture('texture.jpg')

        mesh.textures['texture'] = texture
        mesh.texture_map_to_plane(inplace=True)

        p = pv.Plotter()
        p.add_mesh(mesh)

        p.export_obj("static/generations/generation_" + sys.argv[2] + ".obj")

generate(sys.argv[1])