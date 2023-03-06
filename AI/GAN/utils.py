import scipy.ndimage as nd
import scipy.io as io
import matplotlib
import params

if params.device.type != 'cpu':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
import torch
import os


def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)['instance']
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    elif cube_len == 64:
        voxels = io.loadmat(path)['instance'] 
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

    return voxels

def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.show()


class ShapeNetDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.listdir = os.listdir(self.root)

        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) 

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, params.cube_len), dtype=np.float32)

        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def generateZ(batch):
    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z