import open3d as o3d
import trimesh
import numpy as np

mesh = trimesh.load("dataset/3bd437d38068f4a61f285be552b78f9a.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=40960)
point_cloud = np.array(vertices, dtype=np.float32)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.estimate_normals()


radii = [0.005, 0.01]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

o3d.visualization.draw_geometries([mesh, pcd])