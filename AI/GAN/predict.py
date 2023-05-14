import torch
from model import Generator
from text_to_vec import *

import numpy as np
import pyvista as pv

Generator = Generator().cuda()

model_path = "400.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("floor lamp that is average size and average height"))).astype(np.float64)).reshape(1,512, 1).repeat(16, 1, 1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    pv.PolyData(vertices).plot()