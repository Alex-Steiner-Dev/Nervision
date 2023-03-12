import torch
from utils import *
from model import net_G
import pyvista as pv
import numpy as np
import params

def voxel_to_point_cloud(voxels):
    reshaped_arr = np.zeros((voxels.shape[1], 3))
    for i in range(3):
        flattened_slice = voxels[i, :, :].flatten()
        reshaped_arr[:, i] = flattened_slice[:voxels.shape[1]]

    return reshaped_arr

def plot_point_cloud(points):
    point_cloud = pv.PolyData(points)
    point_cloud.plot()

def test():
    pretrained_file_path_G =  '../TrainedModels/G.pth'

    G = net_G()

    if not torch.cuda.is_available():
        G.load_state_dict(torch.load(pretrained_file_path_G, map_location={'cuda:0': 'cpu'}))
    else:
        G.load_state_dict(torch.load(pretrained_file_path_G))

    G.to(params.device)

    z = generateZ(1)

    fake = G(z)
    samples = fake.unsqueeze(dim=0).detach().cpu().numpy()
    points = voxel_to_point_cloud(samples.reshape(32,32,32))
    plot_point_cloud(points)


test()