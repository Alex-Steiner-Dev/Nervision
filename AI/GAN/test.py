import open3d as o3d
import trimesh
import numpy as np
import pyvista as pv

mesh = trimesh.load("dataset/e71d05f223d527a5f91663a74ccd2338.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=50000)
point_cloud_array = np.array(vertices, dtype=np.float16)

from sklearn.cluster import MiniBatchKMeans


def reduce_dimension(point_cloud, num_points):
    kmeans = MiniBatchKMeans(n_clusters=num_points, random_state=0)
    kmeans.fit(point_cloud)

    cluster_centers = kmeans.cluster_centers_

    return cluster_centers


reduced_points = reduce_dimension(point_cloud_array, num_points=2048)

print(reduced_points.shape)


pv.PolyData(reduced_points).delaunay_3d(.02).extract_geometry().plot()