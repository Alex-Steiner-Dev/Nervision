import trimesh
import numpy as np
import open3d as o3d

# Load the OBJ file
mesh = trimesh.load("moder_chair.obj")

# Extract vertices
vertices = mesh.vertices

# Save the point cloud
np.savetxt("output_point_cloud.txt", vertices)


# Load the point cloud
points = np.loadtxt("output_point_cloud.txt")

# Convert the point cloud to an Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])