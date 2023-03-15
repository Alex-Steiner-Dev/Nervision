import numpy as np
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Dense, Conv1D, Conv1DTranspose, Reshape
from keras.layers.activation import LeakyReLU
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.layers import Input, BatchNormalization
from keras.models import Model
import trimesh
from keras.layers import Dense, Conv2D, Flatten, Reshape, Dropout
from keras.layers.activation import LeakyReLU
from keras.models import Sequential
import matplotlib.pyplot as plt

def show_images(noise, size_fig):
    point_clouds = generator.predict(noise)   
    fig = plt.figure(figsize=size_fig)
    
    for i, pc in enumerate(point_clouds):
        ax = fig.add_subplot(size_fig[0], size_fig[1], i+1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=2)
        ax.set_axis_off()
    
    plt.tight_layout()   

    plt.show()
    
np.random.seed(10)  

noise_dim = 100  
batch_size = 1
steps_per_epoch = 3750  
epochs = 10      

optimizer = Adam(0.0002, 0.5)

x_train = trimesh.load("data/chair_0001.off").sample(2048).reshape(1,2048,3)

def create_generator_cgan():
    generator = Sequential()
    num_points = 2048  
    latent_dim = 100  
    
    generator.add(Dense(256, input_dim=latent_dim))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization())

    generator.add(Dense(num_points * 3, activation='tanh'))
    generator.add(Reshape((num_points, 3)))

    return generator

def create_discriminator_cgan():
    discriminator = Sequential()
    num_points = 2048  
    
    discriminator.add(Conv1D(64, 1, input_shape=(num_points, 3)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(BatchNormalization())

    discriminator.add(Conv1D(128, 1))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(BatchNormalization())

    discriminator.add(Conv1D(256, 1))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(BatchNormalization())

    discriminator.add(Conv1D(512, 1))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(BatchNormalization())

    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    return discriminator


discriminator = create_discriminator_cgan()
generator = create_generator_cgan()

discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
discriminator.trainable = False

gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise).reshape(1,2048,3)

        real_x = x_train

        x = np.concatenate((real_x, fake_x))

        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9

        d_loss = discriminator.train_on_batch(x, disc_y)

        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f'Epoch: {epoch + 1} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
    noise = np.random.normal(0, 1, size=(25, noise_dim))
    show_images(noise, (5, 5))