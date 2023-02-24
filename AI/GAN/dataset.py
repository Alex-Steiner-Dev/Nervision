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

        point_cloud = trimesh.load(train_files[0]).sample(4096)
        objects.append(point_cloud)

    return objects