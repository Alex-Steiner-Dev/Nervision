import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, BatchNormalization, Activation, LeakyReLU, Flatten, Dropout
from keras.optimizers import Adam
from dataset import parse_dataset
import pyvista as pv
from tqdm import tqdm

data = parse_dataset()
epochs = 10000

def discriminator():
    input = Input(shape=(4096,3)) 

    x = Flatten()(input)
    x = Dropout(0.4)(x)

    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(input, output)

    return model

def generator():
    model = Sequential()

    model.add(Dense(512, input_shape=(100,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Dense(4096 * 3, activation='tanh'))  # remove UpSampling1D

    model.add(Reshape((4096, 3)))  # change to output shape to (None, 4096, 3)

    return model

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

    discr.compile(optimizer='adam', loss='binary_crossentropy')
    
    gan_model = gan(discr, gener)

    for epoch in range(epochs):
        real = data[0][0]
        real = np.reshape(real, (1, real.shape[0], real.shape[1]))

        noise = np.random.normal(0, 1, size=(1, 100))
        fake = gener.predict(noise)

        d_loss_real = discr.train_on_batch(real, np.array([1]))
        d_loss_fake = discr.train_on_batch(fake, np.array([1]))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (4096, 3))
        g_loss = gan_model.train_on_batch(noise, np.array([1]))

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

train()