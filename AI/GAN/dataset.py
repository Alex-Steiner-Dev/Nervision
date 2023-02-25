import trimesh
import glob
import numpy as np

DATA_DIR = "../Data/*"

def parse_dataset():
    print("Loading dataset...")
    
    objects = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        train_files = glob.glob(folder + "/train/*.off")

        point_cloud = trimesh.load(train_files[0]).sample(512)

        voxel = np.zeros((512, 512, 512), dtype=np.float16)

        for i in range(512):
            x, y, z = tuple(map(int, point_cloud[i]))
            voxel[x][y][z] = i   

        objects.append(voxel)

    return objects