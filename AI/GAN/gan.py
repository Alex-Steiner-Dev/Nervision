import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, BatchNormalization, Activation, LeakyReLU, Flatten, Dropout, Conv1DTranspose
from keras.optimizers import Adam
from dataset import parse_dataset
import pyvista as pv
from tqdm import tqdm

data = parse_dataset()
epoch = 100

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
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy", metrics=["accuracy"])

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
    
    points = []

    real = data[0][0]
    noise = np.random.normal(0, 1, size=(1024, 3))
  
    final = np.vstack((real, noise))
    final = final.reshape(1, 2048, 3)
        
    dloss = discr.fit(data, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), epochs=100, batch_size=32)
    #gloss = gan_model.fit(np.random.normal(0, 1, size=(2048, 3)), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), epochs=100, batch_size=32)
      
    points.append(final)

train()