import numpy as np
import trimesh
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
import pyvista as pv
from tensorflow.keras.models import load_model

model = load_model("../Train/point_cloud_classifier.h5")

fake_point_clouds = np.random.rand(2048, 3)
real_point_clouds = trimesh.load("../Data/toilet/test/toilet_0346.off").sample(2048)

fake_point_clouds = np.expand_dims(fake_point_clouds, axis=0)
real_point_clouds = np.expand_dims(real_point_clouds, axis=0)

fake_scores = model.predict(fake_point_clouds)
real_scores = model.predict(real_point_clouds)

print("Score for fake point clouds:\n")
print(fake_scores)

print("Score for real point clouds:\n")
print(real_scores)
