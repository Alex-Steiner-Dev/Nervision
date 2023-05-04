import torch
from model import Generator
import open3d as o3d
from text_to_vec import *
import time

import pyvista as pv
import numpy as np

Generator = Generator().cuda()

model_path = "150.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("lawn chair that is very tall and square and small"))).astype(np.float64)).reshape(1,1,512).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    points = sample.numpy().reshape(2048,3)

    mesh = pv.PolyData(points).plot()
    