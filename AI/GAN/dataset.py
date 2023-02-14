import os
import glob
import trimesh
import numpy as np

DATA_DIR = "../Data/*"

def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("Processing class: {}".format(os.path.basename(folder)))
    
        train_files = glob.glob(folder + "/train/*")
        test_files = glob.glob(folder + "/test/*")

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
    )

