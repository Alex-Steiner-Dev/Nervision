import numpy as np
import keras
import open3d
import trimesh
from keras.layers import Dense, Input, Lambda
from keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')
points = chair_mesh.sample(4096)
points = points.reshape(4096, 3)
text_inputs = [    "modern leather chair",    "traditional wooden chair",    "ergonomic mesh chair",    "leather executive chair",    "fabric dining chair",    "wooden rocking chair",    "modern ergonomic chair",    "traditional leather armchair",    "ergonomic gaming chair",    "leather recliner chair"]


num_points = 4096
latent_dim = 128
pointcloud_input = Input(shape=(num_points, 3))
text_input = Input(shape=(100,))

x = keras.layers.Flatten()(pointcloud_input)

x = keras.layers.Concatenate()([x, text_input])

x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(128, activation='relu')(x)
z_mean = keras.layers.Dense(latent_dim)(x)
z_log_var = keras.layers.Dense(latent_dim)(x)

z = z_mean + keras.backend.exp(0.5 * z_log_var) * keras.backend.random_normal(shape=(latent_dim,))

x = keras.layers.Dense(128, activation='relu')(z)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dense(num_points * 3)(x)
decoder_output = keras.layers.Reshape((num_points, 3))(x)

cvae_model = keras.Model(inputs=[pointcloud_input, text_input], outputs=decoder_output)

def vae_loss(pointcloud_input, decoder_output):
    reconstruction_loss = keras.losses.mean_squared_error(pointcloud_input, decoder_output)
    reconstruction_loss *= num_points * 3
    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return keras.backend.mean(reconstruction_loss + kl_loss)

cvae_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=vae_loss)

cvae_model.fit([points, text_inputs], epochs=500, batch_size=1)

new_text_input_encoded = np.array(["modern gaming chair"])
new_latent_vector = np.random.normal(size=(1, latent_dim))
new_latent_vector[0, :len(new_text_input_encoded)] = new_text_input_encoded
generated_pointcloud = cvae_model.predict([new_latent_vector, np.array([new_text_input_encoded])])

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(generated_pointcloud[0])
open3d.visualization.draw_geometries([pcd])