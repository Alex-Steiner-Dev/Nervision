import torch
import os
import sys

sys.path.append("../AI/GAN/")

from mesh_generation import generate_mesh
from text_to_vec import *

from PIL import Image
from min_dalle import MinDalle

import numpy as np
import pyvista as pv

from model import Generator
import open3d as o3d

import random

Generator = Generator().cuda()

model_path = "../AI/TrainedModels/model.pt" 

checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

model = MinDalle(
    dtype=torch.float32, 
    device = 'cuda',
    is_mega = True, 
    is_reusable=True
)

def generate(text):
    z = torch.from_numpy(text_to_vec(process_text(correct_prompt(text))) + np.random.normal(0, 0.01, 512).astype(np.float64)).reshape(1,1,512).cuda().float()

    progressive_outputs = False
    seamless = True
    grid_size = 1
    temperature = 2
    supercondition_factor = 16
    top_k = 128

    image_stream = model.generate_image_stream(
        text="4k check texture",
        seed=random.randint(0,768),
        grid_size = grid_size,
        progressive_outputs = progressive_outputs,
        is_seamless = seamless,
        temperature=temperature,
        top_k = int(top_k),
        supercondition_factor = supercondition_factor,
    )

    for i in image_stream:
        i.save("texture.jpg")

    with torch.no_grad():
        sample = Generator(z).cpu()

        points = sample.numpy().reshape(2048,3)

        mesh = generate_mesh(points)

        os.mkdir("static/generations/" + sys.argv[2])

        o3d.io.write_triangle_mesh("static/generations/" + sys.argv[2] + "/model.obj", mesh)

        mesh = pv.read("static/generations/" + sys.argv[2] + "/model.obj")
        texture = pv.read_texture('texture.jpg')

        mesh.textures['texture'] = texture
        mesh.texture_map_to_plane(inplace=True)

        p = pv.Plotter()
        p.add_mesh(mesh)

        p.export_gltf("static/generations/" + sys.argv[2] + "/model.gltf")
        p.show()

generate(sys.argv[1])