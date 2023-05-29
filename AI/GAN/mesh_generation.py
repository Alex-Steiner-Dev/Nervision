import open3d as o3d
import numpy as np
import trimesh

def generate_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.04
    )

    mesh = mesh.filter_smooth_simple(number_of_iterations=10)

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

    return mesh