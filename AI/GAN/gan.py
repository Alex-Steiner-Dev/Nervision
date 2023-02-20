import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataset import parse_dataset
import random

data = parse_dataset()

num_points = 4096

def generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(num_points * 3, activation='tanh'))
    model.add(Reshape((num_points, 3)))
    noise = Input(shape=(100,))
    cloud = model(noise)
    return Model(inputs=noise, outputs=cloud)

def discriminator():
    model = Sequential()
    model.add(Conv1D(64, 4, strides=2, padding='same', input_shape=(num_points, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(128, 4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(256, 4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(512, 4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))
    cloud = Input(shape=(num_points, 3))
    validity = model(cloud)
    return Model(inputs=cloud, outputs=validity)

optimizer = Adam(lr=0.0002, beta_1=0.5)

discriminator = discriminator()
discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

generator = generator()

z = Input(shape=(100,))
cloud = generator(z)

discriminator.trainable = False

validity = discriminator(cloud)

combined = Model(z, validity)
combined.compile(loss='mse', optimizer=optimizer)

epochs = 10000
batch_size = 1
save_interval = 1000

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    point_clouds = data[0][random.randint(0, len(data[0]) - 1)]
    point_clouds = point_clouds.reshape(1,4096, 3)

    noise = np.random.normal(0, 1, (batch_size, 100))
    gen_point_clouds = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(point_clouds, valid)
    d_loss_fake = discriminator.train_on_batch(gen_point_clouds, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    g_loss = combined.train_on_batch(noise, valid)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

    if epoch % save_interval == 0:
        noise = np.random.normal(0, 1, (1, 100))
        gen_point_cloud = generator.predict(noise)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(gen_point_cloud[0, :, 0], gen_point_cloud[0, :, 1], gen_point_cloud[0, :, 2])
        plt.savefig(f"generations/generated_point_cloud_{epoch}.png")
        plt.close(fig)

gen_point_cloud.save("model.h5")