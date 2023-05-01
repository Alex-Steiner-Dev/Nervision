import torch
import os
import sys

sys.path.append("../AI/GAN/")

from text_to_vec import *

from PIL import Image
from min_dalle import MinDalle

import numpy as np
import pyvista as pv

from model import Generator
from mesh_generation import *

import cv2

import random
import time

start_time = time.time()

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
    z = torch.from_numpy(text_to_vec(process_text(correct_prompt(text)))).reshape(1,1,512).cuda().float()

    progressive_outputs = False
    seamless = True
    grid_size = 1
    temperature = 2
    supercondition_factor = 16
    top_k = 128

    image_stream = model.generate_image_stream(
        text=sys.argv[2] + " texture",
        seed=random.randint(0,768),
        grid_size = grid_size,
        progressive_outputs = progressive_outputs,
        is_seamless = seamless,
        temperature=temperature,
        top_k = int(top_k),
        supercondition_factor = supercondition_factor,
    )

    os.mkdir("static/generations/" + sys.argv[3])

    for i in image_stream:
        i.save("static/generations/" + sys.argv[3] + "/texture.jpg")

    image = cv2.imread("static/generations/" + sys.argv[3] + '/texture.jpg')

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "LapSRN_x8.pb"
    
    sr.readModel(path)
    sr.setModel("lapsrn",8)
    
    result = sr.upsample(image)
 
    cv2.imwrite("static/generations/" + sys.argv[3] + '/texture.jpg', result)

    with torch.no_grad():
        sample = Generator(z).cpu()

        points = sample.numpy().reshape(2048,3)

        mesh = generate_mesh(points)
        o3d.io.write_triangle_mesh("static/generations/" + sys.argv[3] + "/model.obj", mesh)

        mesh = pv.read("static/generations/" + sys.argv[3] + "/model.obj")

        texture = pv.read_texture("static/generations/" + sys.argv[3] + '/texture.jpg')

        mesh.textures['texture'] = texture
        mesh.texture_map_to_plane(inplace=True)

        p = pv.Plotter()
        p.add_mesh(mesh)

        p.export_gltf("static/generations/" + sys.argv[3] + "/model.gltf")
        p.show()

generate(sys.argv[1])
print((time.time() - start_time))