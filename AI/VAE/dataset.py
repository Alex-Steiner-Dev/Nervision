import sys

sys.path.append("../GAN/")

from params import *
from utils import *
import numpy as np
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import torch
from model import net_G
import numpy as np

DATA_DIR = "../Data/VolumetricData/*"

def load_model():
    generator_path =  '../TrainedModels/G.pth'

    generator = net_G()

    if not torch.cuda.is_available():
        generator.load_state_dict(torch.load(generator_path, map_location={'cuda:0': 'cpu'}))
    else:
        generator.load_state_dict(torch.load(generator_path))

    generator.to(params.device)

    return generator

def predict_noisy():
    generator = load_model()
    
    z = generateZ(1)
    fake = generator(z)
    noisy = fake.unsqueeze(dim=0).detach().cpu().numpy()

    return noisy

def getVoxelsFromMat(path):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    voxels = nd.zoom(voxels, (1, 1, 1), mode='constant', order=0)

    return voxels

def parse_dataset():
    print("Loading dataset...")
    
    voxels = []
    noisy = []
    
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        voxels_files = glob.glob(folder + "/30/train/*.mat")

        if folder == "../Data/VolumetricData\chair":
            for f in voxels_files:
                voxel = getVoxelsFromMat(f)
                voxels.append(voxel)
                noisy.append(predict_noisy())

    print("Done!")

    return voxels, noisy