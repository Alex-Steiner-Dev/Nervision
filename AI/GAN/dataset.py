import numpy as np
import glob
import numpy as np
import scipy.io as io
import scipy.ndimage as nd

DATA_DIR = "../Data/VolumetricData/*"

def getVoxelsFromMat(path):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)

    return voxels

def parse_dataset():
    print("Loading dataset...")
    
    voxels = []
    labels = []
    
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        voxels_files = glob.glob(folder + "/30/train/*.mat")
        label_files = glob.glob(folder + "/30/train/*.txt")

        if folder == "../Data/VolumetricData/chair":
            for f in voxels_files:
                voxel = getVoxelsFromMat(f)
                voxels.append(voxel)

            for f in label_files:
                with open(label_files) as text_file:
                    labels.append(text_file.readlines())

    print("Done!")

    return voxels
