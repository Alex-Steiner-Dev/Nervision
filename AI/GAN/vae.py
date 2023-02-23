import numpy as np
from keras.layers import Input, Dense, Lambda, Reshape, Flatten
from keras.models import Model
from keras import backend as K
import trimesh
import keras

chair_meshes = trimesh.load("../Data/chair/train/chair_0001.off")
chair_point_clouds = chair_meshes.sample(2048)
chair_point_clouds = chair_point_clouds.reshape(1, 2048, 3)

latent_dim = 2

def sampling(args):
    """Reparameterization trick to sample from the latent space"""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

input_shape = (2048, 3)
inputs = Input(shape=input_shape, name='encoder_input')
x = Flatten()(inputs)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(256, activation='relu')(latent_inputs)
x = Dense(512, activation='relu')(x)
x = Dense(2048*3, activation='sigmoid')(x)
outputs = Reshape((2048, 3))(x)

decoder = Model(latent_inputs, outputs, name='decoder')

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

reconstruction_loss = keras.losses.binary_crossentropy(Flatten()(inputs), Flatten()(outputs))
reconstruction_loss *= 2048*3
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(chair_point_clouds, epochs=50, batch_size=1)

# Generate new point clouds from the trained VAE model
z_sample = np.random.normal(size=(2048, latent_dim))
x_decoded = decoder.predict(z_sample)

mesh = trimesh.Trimesh(vertices=x_decoded[0], faces=None)
mesh = mesh.fix_normals()
#mesh = mesh.fill_holes()
#mesh = mesh.compute_convex_hull()
#mesh = mesh.split()[0]
#mesh = mesh.simplify(0.01)
#trimesh.repair.fill_holes(mesh)
#trimesh.repair.fix_inversion(mesh)
#trimesh.repair.fix_normals(mesh)

# Export generated mesh as an OBJ file
#trimesh.exchange.export.export_mesh(mesh, 'generated_chair_mesh.obj')