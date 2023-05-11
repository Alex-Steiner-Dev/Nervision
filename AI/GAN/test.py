import open3d as o3d
import trimesh
import numpy as np

mesh = trimesh.load("dataset/40f1be4ede6113a2e03aea0698586c31.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=2048)
point_cloud = np.array(vertices, dtype=np.float32)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.estimate_normals()

o3d.visualization.draw_geometries([pcd])