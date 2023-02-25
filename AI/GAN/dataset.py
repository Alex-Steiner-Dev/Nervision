import numpy as np
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd

DATA_DIR = "volumetric_data/*"

def getVoxelsFromMat(path):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    voxels = nd.zoom(voxels, (1, 1, 1), mode='constant', order=0)

    return voxels

def parse_dataset(box_size=32):
    print("Loading dataset...")
    
    objects = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        train_files = glob.glob(folder + "/30/train/*.mat")

        if folder == "volumetric_data\chair":
            for f in train_files:
                voxels = getVoxelsFromMat(f)

                objects.append(voxels)

    print("Done!")

    return objects