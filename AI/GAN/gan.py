import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "../Data/*"

def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("Processing class: {}".format(os.path.basename(folder)))
    
        train_files = glob.glob(folder + "/train/*")

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

    return (
        np.array(train_points),
        np.array(train_labels),
    )

train_point, train_labels = parse_dataset()

print(train_labels)