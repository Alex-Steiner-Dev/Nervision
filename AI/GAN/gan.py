import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input, BatchNormalization, Activation, LeakyReLU
from keras.optimizers import Adam
from dataset import parse_dataset
import pyvista as pv

#data = parse_dataset()
input_shape = (2048, 3)
epoch = 100

def build_generator():
    model = Sequential([
        Dense(3,input_shape=input_shape, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='tanh'),

    ])

    return model

def build_discriminator():
    model = Sequential([
        Dense(3,input_shape=input_shape)
    ])

    return model

def train():
    generator = build_generator()
    discriminator = build_discriminator()

    z = np.random.rand(2048, 3)

    generation = generator.predict(z)

    point_cloud = pv.PolyData(generation)
    point_cloud.plot()

    ouput = discriminator.predict(generation)

    for i in range(epoch):
        pass

train()