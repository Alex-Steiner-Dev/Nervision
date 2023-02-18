import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, BatchNormalization, Activation, LeakyReLU, Flatten, Dropout, Conv1DTranspose
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
    model = Sequential()

    model.add(Dense(256, input_shape=(100,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(2048*3, activation='tanh'))
    model.add(Reshape((2048, 3)))

    noise = Input(shape=(100,))
    point_cloud = model(noise)

    return Model(noise, point_cloud)

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

    noise = np.random.normal(0, 1, size=(1, 100))

    z = gener.predict(noise)
    z = z.reshape((2048,3))

    point_cloud = pv.PolyData(z)
    point_cloud.plot()

    for i in range(100):
        pass

train()