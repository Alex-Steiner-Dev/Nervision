from keras.models import load_model
import numpy as np
import open3d as o3d
import trimesh

decoder = load_model("vae.h5")

latent_vectors = np.random.normal(size=(5000, 128))
generated_points = decoder.predict(latent_vectors)

mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(generated_points)

vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

mesh.export("reconstructed_mesh.obj")