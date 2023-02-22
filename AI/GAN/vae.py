import trimesh
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, Flatten, Dense, Reshape, Conv3DTranspose, Lambda
import keras.backend as K
import keras
import trimesh
import open3d as o3d

# Load the chair mesh using Trimesh
chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')

# Convert the mesh to a point cloud using PyVista
point_cloud = chair_mesh.sample(10000)

# Define the Keras VAE model
latent_dim = 128

encoder_inputs = keras.Input(shape=(point_cloud.shape[1],))
x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
x = keras.layers.Dense(128, activation='relu')(x)
z_mean = keras.layers.Dense(latent_dim)(x)
z_log_var = keras.layers.Dense(latent_dim)(x)

z = z_mean + keras.backend.exp(0.5 * z_log_var) * keras.backend.random_normal(shape=(latent_dim,))

decoder_inputs = keras.Input(shape=(latent_dim,))
x = keras.layers.Dense(128, activation='relu')(decoder_inputs)
x = keras.layers.Dense(256, activation='relu')(x)
decoder_outputs = keras.layers.Dense(point_cloud.shape[1])(x)

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = keras.Model(decoder_inputs, decoder_outputs, name='decoder')

vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name='vae')

# Train the VAE on the point cloud
vae.compile(optimizer=keras.optimizers.Adam(lr=0.001),
            loss=keras.losses.mean_squared_error)
vae.fit(point_cloud, point_cloud, epochs=50, batch_size=32)

# Generate a new set of latent vectors
latent_vectors = np.random.normal(size=(1, latent_dim))

# Use the VAE decoder to generate a new set of points
generated_points = decoder.predict(latent_vectors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])