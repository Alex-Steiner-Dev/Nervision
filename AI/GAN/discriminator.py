import numpy as np
import trimesh
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
import pyvista as pv

input_shape = (2048,3)

inputs = Input(shape=input_shape)

x = Dense(512, activation=LeakyReLU(alpha=0.2))(inputs)
x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
x = Dense(128, activation=LeakyReLU(alpha=0.2))(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam')

fake_point_clouds = np.random.rand(2048, 3)
real_point_clouds = trimesh.load("../Data/toilet/test/toilet_0346.off").sample(2048)

point_cloud = pv.PolyData(real_point_clouds)
point_cloud.plot()

fake_point_clouds = np.expand_dims(fake_point_clouds, axis=0)
real_point_clouds = np.expand_dims(real_point_clouds, axis=0)

print(fake_point_clouds.shape)
print(real_point_clouds.shape)

fake_scores = model.predict(fake_point_clouds)
real_scores = model.predict(real_point_clouds)

print("Average score for fake point clouds:", np.mean(fake_scores))
print("Average score for real point clouds:", np.mean(real_scores))
