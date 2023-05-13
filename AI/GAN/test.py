import open3d as o3d
import trimesh
import numpy as np
import umap

mesh = trimesh.load("dataset/40f1be4ede6113a2e03aea0698586c31.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=5000)
point_cloud_array = np.array(vertices, dtype=np.float16)

umap_obj = umap.UMAP(n_components=3, n_neighbors=10)

# Step 5: Fit and transform the data
lower_dim_representation = umap_obj.fit_transform(point_cloud_array)

# Step 7: Inverse transform
original_dimension_representation = umap_obj.inverse_transform(lower_dim_representation)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(original_dimension_representation)
pcd.estimate_normals()

o3d.visualization.draw_geometries([pcd])