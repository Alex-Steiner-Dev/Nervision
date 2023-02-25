import trimesh
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd

DATA_DIR = "volumetric_data/*"

def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (1, 1, 1), mode='constant', order=0)
    return voxels

def parse_dataset():
    print("Loading dataset...")
    
    objects = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        train_files = glob.glob(folder + "/30/train/*.mat")

        voxels = getVoxelsFromMat(train_files[0])

        objects.append(voxels)

    return objects