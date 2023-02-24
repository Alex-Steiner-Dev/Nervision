<<<<<<< HEAD
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Flatten, BatchNormalization, Lambda, Reshape, Activation, LeakyReLU
from keras.activations import selu
from keras.layers import Multiply, Add
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

latent_dim = 4096

###### Encoder ######
encoder_input = Input(shape=(1,4096, 3))

x = Conv2D(32, kernel_size=5, activation = LeakyReLU(0.02), strides = 1, padding = 'same')(encoder_input)
x = BatchNormalization()(x)

filter_size = [64,128,256,512, 1024, 2048, 4096]
for i in filter_size:
    x = Conv2D(i, kernel_size=5, activation = LeakyReLU(0.02), strides = 2, padding = 'same')(x)
    x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(4096*2, activation = selu)(x)
encoder_output = BatchNormalization()(x)

# sampling layer
mu = Dense(latent_dim)(encoder_output)
log_var = Dense(latent_dim)(encoder_output)

epsilon = K.random_normal(shape = (tf.shape(mu)[0], tf.shape(mu)[1]))
sigma = tf.exp(0.5 * log_var)
=======
import numpy as np
import keras
import open3d
import trimesh

# Load the chair mesh using Trimesh
chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')
points = chair_mesh.sample(5000)

points = (points - points.mean(axis=0)) / points.std(axis=0) # normalize the points

# Define the encoder and decoder architectures
latent_dim = 128

encoder_inputs = keras.Input(shape=(points.shape[1],))
x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
x = keras.layers.Dense(128, activation='relu')(x)
z_mean = keras.layers.Dense(latent_dim)(x)
z_log_var = keras.layers.Dense(latent_dim)(x)
>>>>>>> parent of 6769dd67c (VAE text test)

z_eps = Multiply()([sigma, epsilon])
z = Add()([mu, z_eps])

<<<<<<< HEAD
encoder = Model(encoder_input, outputs = [mu, log_var, z], name = 'encoder')
encoder.summary()

###### Decoder ######
decoder = Sequential()
decoder.add(Dense(1024, activation = selu, input_shape = (latent_dim, )))
decoder.add(BatchNormalization())

decoder.add(Dense(8192, activation = selu))
decoder.add(Reshape((4,4,512)))

decoder.add(Conv2DTranspose(256, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(128, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(64, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(32, (5,5), activation = LeakyReLU(0.02), strides = 2, padding = 'same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(3, (5,5), activation = "sigmoid", strides = 1, padding = 'same'))
decoder.add(BatchNormalization())

decoder.summary()

def reconstruction_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def kl_loss(mu, log_var):
    loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return loss

def vae_loss(y_true, y_pred, mu, log_var):
    return reconstruction_loss(y_true, y_pred) + (1 / (4096*3)) * kl_loss(mu, log_var)
=======
decoder_inputs = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(128, activation='relu')(decoder_inputs)
x = keras.layers.Dense(256, activation='relu')(x)
decoder_outputs = keras.layers.Dense(points.shape[1])(x)

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name='vae')

# Define the loss function
reconstruction_loss = keras.losses.mean_squared_error(encoder_inputs, vae_outputs)
reconstruction_loss *= points.shape[1]
kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
kl_loss = keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Train the VAE
vae.compile(optimizer=keras.optimizers.Adam(lr=0.001))
vae.fit(points, epochs=1000, batch_size=1)

vae.save_weights("vae.h5")

# Generate new point clouds from the VAE
latent_vectors = np.random.normal(size=(1000, latent_dim))
generated_points = decoder.predict(latent_vectors)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(generated_points)

# Visualize the point cloud
open3d.visualization.draw_geometries([pcd])
>>>>>>> parent of 6769dd67c (VAE text test)
