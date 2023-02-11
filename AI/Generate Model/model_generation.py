import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(1234)

# The same pre-processing steps as in the classification model
def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        class_map[i] = folder.split("/")[-1]
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
        class_map,
    )

# Input shape for the generator and discriminator models
input_shape = (2048, 3)

# Define the generator
def generator():
    inputs = keras.Input(shape=input_shape)

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dense(2048 * 3, activation="tanh")(x)
    outputs = layers.Reshape(input_shape)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="generator")

# Define the discriminator
def discriminator():
    inputs = keras.Input(shape=input_shape)

    x = layers.Dense(1024, activation="relu")(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="discriminator")

# Generate random noise to feed the generator
def get_noise(batch_size):
    return np.random.rand(batch_size, 100)

# Compile the discriminator
