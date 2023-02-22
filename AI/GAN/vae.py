import numpy as np
import keras
import open3d
import trimesh

# Load the chair mesh using Trimesh
chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')
points = chair_mesh.sample(1000000)


points = (points - points.mean(axis=0)) / points.std(axis=0) # normalize the points

# Define the encoder and decoder architectures
latent_dim = 128

encoder_inputs = keras.Input(shape=(points.shape[1],))
x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
x = keras.layers.Dense(128, activation='relu')(x)
z_mean = keras.layers.Dense(latent_dim)(x)
z_log_var = keras.layers.Dense(latent_dim)(x)

z = z_mean + keras.backend.exp(0.5 * z_log_var) * keras.backend.random_normal(shape=(latent_dim,))

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
vae.fit(points, epochs=10000, batch_size=1)

vae.save("vae.h5")

# Generate new point clouds from the VAE
latent_vectors = np.random.normal(size=(16384, latent_dim))
generated_points = decoder.predict(latent_vectors)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(generated_points)

# Visualize the point cloud
open3d.visualization.draw_geometries([pcd])