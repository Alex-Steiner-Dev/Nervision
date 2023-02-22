from keras.models import load_model
import numpy as np
import open3d

decoder = load_model("vae.h5")

latent_vectors = np.random.normal(size=(5000, 128))
generated_points = decoder.predict(latent_vectors)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(generated_points)

# Visualize the point cloud
open3d.visualization.draw_geometries([pcd])