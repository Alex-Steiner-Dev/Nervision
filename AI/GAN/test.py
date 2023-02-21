import os
import numpy as np
from keras.layers import Input, LeakyReLU, BatchNormalization
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.core import Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.ndimage as nd

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

    gen_model = Model(inputs=input_layer, outputs=a)

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
    a = BatchNormalization()(a, training=True)
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

    dis_model = Model(inputs=dis_input_layer, outputs=a)

    return dis_model

def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)
    plt.close()

gen_learning_rate = 0.0025
dis_learning_rate = 0.00001
beta = 0.5
batch_size = 32
z_size = 200
DIR_PATH = 'volumetric_data/'
generated_volumes_dir = 'Models'

generator = build_generator()
discriminator = build_discriminator()

gen_optimizer = Adam(learning_rate=gen_learning_rate, beta_1=beta)
dis_optimizer = Adam(learning_rate=dis_learning_rate, beta_1=0.9)

generator.compile(loss="binary_crossentropy", optimizer="adam")
discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

discriminator.trainable = False
adversarial_model = Sequential()
adversarial_model.add(generator)
adversarial_model.add(discriminator)
adversarial_model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=gen_learning_rate, beta_1=beta))

def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def get3ImagesForACategory(obj='airplane', train=True, cube_len=64, obj_ratio=1.0):
    obj_path = DIR_PATH + obj + '/30/'
    obj_path += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio * len(fileList))]
    volumeBatch = np.asarray([getVoxelsFromMat(obj_path + f, cube_len) for f in fileList], dtype=bool)
    return volumeBatch

volumes = get3ImagesForACategory(obj='airplane', train=True, obj_ratio=1.0)
volumes = volumes[..., np.newaxis].astype(float)

for epoch in range(1000):
    print("Epoch:", epoch)

    gen_losses = []
    dis_losses = []

    number_of_batches = int(volumes.shape[0] / batch_size)
    print("Number of batches:", number_of_batches)
    for index in range(number_of_batches):
        print("Batch:", index + 1)

        z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

        gen_volumes = generator.predict(z_sample,verbose=3)

        discriminator.trainable = True

        labels_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
        labels_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))
             
        loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
        loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

        d_loss = 0.5 * (loss_real + loss_fake)

        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)

        g_loss = adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))

        gen_losses.append(g_loss)
        dis_losses.append(d_loss)

        if index % 10 == 0:
            z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            generated_volumes = generator.predict(z_sample2, verbose=3)
            for i, generated_volume in enumerate(generated_volumes[:5]):
                voxels = np.squeeze(generated_volume)
                voxels[voxels < 0.5] = 0.
                voxels[voxels >= 0.5] = 1.
                saveFromVoxels(voxels, "Predictions/img_{}_{}_{}".format(epoch, index, i))

generator.save_weights(os.path.join(generated_volumes_dir, "generator_weights.h5"))
discriminator.save_weights(os.path.join(generated_volumes_dir, "discriminator_weights.h5"))