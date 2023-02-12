import numpy as np
import keras
from keras import layers
import tensorflow as tf
import pyvista as pv

def make_discriminator():
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=3, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def make_generator():
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=3, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='tanh'))
    return model

discriminator = make_discriminator()
generator = make_generator()


combined_model = keras.Sequential([generator, discriminator])
combined_model.compile(loss='binary_crossentropy', optimizer='adam')

noise = np.random.rand(2048, 3)

generated_points = generator.predict(noise)

labels = discriminator.predict(generated_points)

point_cloud = pv.PolyData(generated_points)
point_cloud.plot()