import os
import glob
import trimesh
import numpy as np

DATA_DIR = "../Data/*"

def parse_dataset(num_points=4096):
    objects = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        train_points = []
    
        train_files = glob.glob(folder + "/train/*")

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))

        objects.append(train_points)

    return objects