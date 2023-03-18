import torch
import pyvista as pv
from model import Generator
import numpy as np

Generator = Generator(num_points=2048).cuda()

model_path = "850.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.randn(1, 1, 128).cuda()

with torch.no_grad():
    sample = Generator(z).cpu()
    point_cloud = pv.PolyData(np.array((sample)).reshape(2048,3))
    point_cloud.plot()