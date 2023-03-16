import torch
import numpy as np
import pyvista as pv
import random
from model import WarpingGAN

WarpingGAN = WarpingGAN(num_points=2048).cuda()

model_path = "1950.pt"
checkpoint = torch.load(model_path,map_location='cuda:0')
WarpingGAN.load_state_dict(checkpoint['G_state_dict'])

pointclouds = torch.Tensor([])
z = torch.randn(16, 1, 128).cuda()

with torch.no_grad():
    sample = WarpingGAN(z).cpu()
pointclouds = torch.cat((pointclouds, sample), dim=0)

print(pointclouds.shape)

fake = pointclouds.cpu().numpy()[random.randint(0,15)]
pv.PolyData(fake).plot()