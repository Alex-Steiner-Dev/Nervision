from plyfile import PlyData
import numpy as np
import pyvista as pv
import open3d as o3d

ply_data = PlyData.read("file.ply")
points = ply_data['vertex']
points = np.vstack([points['x'], points['y'], points['z']]).T

pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))

pcd.estimate_normals()

radius = 0.2 
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

o3d.visualization.draw_geometries([mesh])