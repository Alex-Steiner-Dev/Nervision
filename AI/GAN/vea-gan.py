# Import libraries
import keras
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from dataset import parse_dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Define hyperparameters
input_shape = (4096, 3)
latent_dim = 3
epochs = 1000

# Load dataset
dataset = parse_dataset()[0].reshape(1,4096,3)

# Define VAE architecture
input_layer = Input(shape=input_shape)
hidden_layer = Dense(512, activation='relu')(input_layer)
z_mean = Dense(latent_dim)(hidden_layer)
z_log_var = Dense(latent_dim)(hidden_layer)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

latent_layer = Lambda(sampling)([z_mean, z_log_var])
hidden_layer = Dense(512, activation='relu')(latent_layer)
output_layer = Dense(np.prod(input_shape[1:]), activation='sigmoid')(hidden_layer)
vae = Model(input_layer, output_layer)

# Define VAE loss function
reconstruction_loss = keras.losses.binary_crossentropy(input_layer, output_layer)
reconstruction_loss *= np.prod(input_shape[1:])
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.Adam(0.0002))

# Define GAN architecture
z_input_layer = Input(shape=(latent_dim,))
hidden_layer = Dense(512, activation='relu')(z_input_layer)
output_layer = Dense(np.prod(input_shape[1:]), activation='sigmoid')(hidden_layer)
generator = Model(z_input_layer, output_layer)

input_layer = Input(shape=input_shape)
hidden_layer = Dense(512, activation='relu')(input_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
discriminator = Model(input_layer, output_layer)

# Define GAN loss function
def gan_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)

# Compile GAN model
discriminator.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss=gan_loss)

discriminator.trainable = False

gan_input_layer = Input(shape=(latent_dim,))
gan_output_layer = discriminator(generator(gan_input_layer))
gan = Model(gan_input_layer, gan_output_layer)

gan.compile(optimizer=keras.optimizers.Adam(0.0002, 0.5), loss=gan_loss)

for epoch in range(epochs):
    # Train VAE
    vae_loss = vae.train_on_batch(dataset, None)

    # Train GAN
    latent_vectors = np.random.normal(size=(dataset.shape[1], latent_dim)).reshape(1, 4096, 3)
    fake_labels = np.zeros((dataset.shape[1], 1))
    gan_loss = gan.train_on_batch(latent_vectors, None)

    # Print progress
    print('Epoch {}/{} - VAE loss: {}, GAN loss: {}'.format(epoch+1, epochs, vae_loss, gan_loss))
