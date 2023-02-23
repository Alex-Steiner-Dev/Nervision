from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import numpy as np

import numpy as np
import trimesh

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')
X_train = chair_mesh.sample(4096)
X_train = X_train.reshape(1, 4096,3)

input_shape = (4096, 3)

# Define the size of the latent space
latent_dim = 32

# Define the encoder network
encoder_input = Input(shape=input_shape)
x = Dense(128, activation='relu')(encoder_input)
x = Dense(64, activation='relu')(x)
mean = Dense(latent_dim)(x)
log_var = Dense(latent_dim)(x)

# Define the sampling function
def sampling(args):
    mean, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mean)[0], latent_dim), mean=0., stddev=1.)
    return mean + K.exp(0.5 * log_var) * epsilon

# Define the latent space sampling layer
latent = Lambda(sampling, output_shape=(latent_dim,))([mean, log_var])

# Define the decoder network
decoder_input = Input(shape=(latent_dim,))
x = Dense(64, activation='relu')(decoder_input)
x = Dense(128, activation='relu')(x)
decoder_output = Dense(3)(x)

# Define the encoder model
encoder = Model(encoder_input, [mean, log_var, latent])

# Define the decoder model
decoder = Model(decoder_input, decoder_output)

# Define the VAE model
vae_output = decoder(latent)
vae = Model(encoder_input, vae_output)

# Define the VAE loss function
def vae_loss(encoder_input, vae_output):
    reconstruction_loss = mse(encoder_input, vae_output)
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)

# Compile the VAE model
vae.compile(optimizer=Adam(learning_rate=0.0001), loss=vae_loss)

# Train the VAE model
vae.fit(X_train, X_train, epochs=10, batch_size=32)