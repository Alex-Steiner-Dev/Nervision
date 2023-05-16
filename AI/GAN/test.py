import trimesh
import trimesh
import numpy as np
import open3d as o3d


mesh = trimesh.load("dataset/e71d05f223d527a5f91663a74ccd2338.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=2048)
point_cloud_array = np.array(vertices, dtype=np.float32)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
o3d.visualization.draw_geometries([pcd])