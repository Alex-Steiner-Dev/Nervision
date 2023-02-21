import os
import numpy as np
from keras.layers import Input, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import Adam
from dataset import parse_dataset
import matplotlib.pyplot as plt

def build_generator():
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5

    input_layer = Input(shape=gen_input_shape)

    a = Conv3DTranspose(filters=gen_filters[0],
                 kernel_size=gen_kernel_sizes[0],
                 strides=gen_strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation='relu')(a)

    for i in range(gen_convolutional_blocks - 1):
        a = Conv3DTranspose(filters=gen_filters[i + 1],
                     kernel_size=gen_kernel_sizes[i + 1],
                     strides=gen_strides[i + 1], padding='same')(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(activation=gen_activations[i + 1])(a)

    gen_model = Model(inputs=[input_layer], outputs=[a])
    return gen_model

def build_discriminator():
    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu',
                       'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5

    dis_input_layer = Input(shape=dis_input_shape)

    a = Conv3D(filters=dis_filters[0],
               kernel_size=dis_kernel_sizes[0],
               strides=dis_strides[0],
               padding=dis_paddings[0])(dis_input_layer)
    a = LeakyReLU(dis_alphas[0])(a)

    for i in range(dis_convolutional_blocks - 1):
        a = Conv3D(filters=dis_filters[i + 1],
                   kernel_size=dis_kernel_sizes[i + 1],
                   strides=dis_strides[i + 1],
                   padding=dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        elif dis_activations[i + 1] == 'sigmoid':
            a = Activation(activation='sigmoid')(a)

    dis_model = Model(inputs=[dis_input_layer], outputs=[a])
    return dis_model

def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    gen_learning_rate = 0.0025
    dis_learning_rate = 10e-5
    beta = 0.5
    batch_size = 1
    z_size = 200
    epochs = 1000

    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    discriminator.trainable = False

    input_layer = Input(shape=(1, 1, 1, z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    adversarial_model = Model(inputs=[input_layer], outputs=[validity])
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    volumes = parse_dataset()[0]
    volumes = volumes.reshape(1, 64, 64, 64)

    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))

    os.system("cls")

    for epoch in range(epochs):
        gen_losses = []
        dis_losses = []

        z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
   
        gen_volumes = generator.predict_on_batch(z_sample)

        discriminator.trainable = True

        loss_real = discriminator.train_on_batch(volumes, labels_real)
        loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

        d_loss = 0.5 * np.add(loss_real, loss_fake)

        discriminator.trainable = False
   
        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        g_loss = adversarial_model.train_on_batch(z, labels_real)
  
        gen_losses.append(g_loss)
        dis_losses.append(d_loss)

        print(f"[Epoch: {epoch}] - [Generator Loss: {g_loss}] - [Discriminator Loss - {d_loss}]")

        if epoch % 15 == 0:
            z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            generated_volumes = generator.predict(z_sample2, verbose=3)
            for i, generated_volume in enumerate(generated_volumes[:5]):
                voxels = np.squeeze(generated_volume)
                voxels[voxels < 0.5] = 0.
                voxels[voxels >= 0.5] = 1.
                saveFromVoxels(voxels, "Predictions/img_{}".format(epoch))
            
    generator.save_weights(os.path.join("Models", "generator_weights.h5"))
    discriminator.save_weights(os.path.join("Models", "discriminator_weights.h5"))