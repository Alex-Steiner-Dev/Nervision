import open3d as o3d
import numpy as np


def point_cloud_to_mesh_obj(points):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(points)

    # Compute the mesh
    mesh, densities = pcd.compute_triangle_mesh()

    # Save the mesh to a file
    o3d.write_triangle_mesh("mesh.ply", mesh)
