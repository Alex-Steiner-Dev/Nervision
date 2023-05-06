import torch
from model import Generator
import open3d as o3d
from text_to_vec import *
import random

import pyvista as pv
import numpy as np

Generator = Generator().cuda()

model_path = "200.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("clock that is average thickness and average size"))).astype(np.float64)).reshape(1,512, 1).repeat(16, 1, 1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    points = sample.numpy()[0]

    cloud = pv.PolyData(points).plot()