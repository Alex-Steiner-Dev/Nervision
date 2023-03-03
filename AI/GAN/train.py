from dataset import parse_dataset
from GAN import *
import numpy as np

x_train, y_train = parse_dataset()
x_train = np.array(x_train)
y_train = np.array(y_train)

import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Reshape, BatchNormalization, Activation, Dropout, Flatten, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Define the generator model
def build_generator():
    model = Sequential()

    model.add(Dense(256, input_dim=100))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2048*3, activation='tanh'))
    model.add(Reshape((2048, 3)))

    noise = Input(shape=(100,))
    points = model(noise)

    return Model(inputs=noise, outputs=points)

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(2048, 3, 1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    points = Input(shape=(2048, 3))
    validity = model(Reshape((2048, 3, 1))(points))

    return Model(points, validity)


# Define the combined model
def build_combined(generator, discriminator):
    z = Input(shape=(100,))
    points = generator(z)
    validity = discriminator(points)
    return Model(inputs=z, outputs=validity)


# Train the PointGAN model to generate a 3D point cloud
generator = build_generator()
discriminator = build_discriminator()
combined = build_combined(generator, discriminator)

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

z = Input(shape=(100,))
points = generator(z)
validity = discriminator(points)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the PointGAN model
batch_size = 32
epochs = 10000
sample_interval = 1000

for epoch in range(epochs):
    # Generate a batch of new points
    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_points = generator.predict(noise)  
    # Select a random batch of points from the target mesh
    idx = np.random.randint(0, x_train, batch_size)
    real_points = x_train[idx]

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_points, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_points, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    if epoch % sample_interval == 0:
        print("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))