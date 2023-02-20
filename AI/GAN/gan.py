import numpy as np
from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataset import parse_dataset
import random

data = parse_dataset()

epochs = 10000
batch_size = 1
save_interval = 1000
latent_dim = 200
im_dim = 64
kernel_size = 4
strides = 2
batch_size = 1

def build_generator():
    model = Sequential()
    model.add(Conv3DTranspose(filters=512, kernel_size=kernel_size,
        strides=(1, 1, 1), kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=256, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=128, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv3DTranspose(filters=64, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3DTranspose(filters=1, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    noise = Input(shape=(1, 1, 1, latent_dim))
    image = model(noise)

    return Model(inputs=noise, outputs=image)

def build_discriminator():
    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv3D(filters=128, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv3D(filters=256, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv3D(filters=512, kernel_size=kernel_size,
        strides=strides, kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv3D(filters=1, kernel_size=kernel_size,
        strides=(1, 1, 1), kernel_initializer='glorot_normal',
        bias_initializer='zeros', padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    image = Input(shape=(im_dim, im_dim, im_dim, 1))
    validity = model(image)

    return Model(inputs=image, outputs=validity)

dis_optim = Adam(lr=1e-5, beta_1=0.5)
gen_optim = Adam(lr=0.0025, beta_1=0.5)

discriminator = build_discriminator()

discriminator.compile(loss='binary_crossentropy', optimizer=dis_optim)

generator = build_generator()

z = Input(shape=(1, 1, 1, latent_dim))
img = generator(z)

discriminator.trainable = False
validity = discriminator(img)

combined = Model(input=z, output=validity)
combined.compile(loss='binary_crossentropy', optimizer=gen_optim)

for epoch in range(epochs):
    real = data[0][0]

    z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1,latent_dim]).astype(np.float32)
    fake = generator.predict(z)

    real = np.expand_dims(real, axis=4)

    lab_real = np.reshape([1] *batch_size, (-1, 1, 1, 1, 1))
    lab_fake = np.reshape([0] *batch_size, (-1, 1, 1, 1, 1))

    d_loss_real = discriminator.train_on_batch(real, lab_real)
    d_loss_fake = discriminator.train_on_batch(fake, lab_fake)

    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

    z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1,latent_dim]).astype(np.float32)

    g_loss = combined.train_on_batch(z, np.reshape([1] *batch_size, (-1, 1, 1, 1, 1))).astype(np.float64)

generator.save("model.h5")