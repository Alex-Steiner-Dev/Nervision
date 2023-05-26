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

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("cocktail table that is tall and square and average size"))).astype(np.float64)).reshape(1,512, 1).repeat(32,1,1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40,std_ratio=4.0)
    pcd = cl

    #mesh = generate_mesh(pcd.points)

    #o3d.visualization.draw_geometries([mesh])


    pv.PolyData(vertices).plot()
