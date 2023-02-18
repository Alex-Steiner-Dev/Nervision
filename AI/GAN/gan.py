import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, BatchNormalization, Activation, LeakyReLU, Flatten, Dropout
from keras.optimizers import Adam
from dataset import parse_dataset
import pyvista as pv

#data = parse_dataset()

def discriminator():
    input = Input(shape=(2048,3))

    x = Flatten()(input)
    x = Dropout(0.4)(x)

    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(input, output)
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    return model

def generator():
    input = Input(shape=(2048))

    x = Dense(256, activation=LeakyReLU(alpha=0.2))(input)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(input)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(input)
    x = Dense(6144, activation=LeakyReLU(alpha=0.2))(input)

    output = Reshape((2048, 3))(x)

    return Model(input, output)

def gan(discr, gener):
    discr.trainable = False

    model = Sequential([
        gener,
        discr
    ])

    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

    return model

def train():
    discr = discriminator()
    gener = generator()

    gan_model = gan(discr, gener)

    z = np.random.rand(2048, 3)

    #point_cloud = pv.PolyData(generation)
    #point_cloud.plot()

    for i in range(100):
        pass

train()