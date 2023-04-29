import open3d as o3d
import numpy as np
import pyvista as pv

def generate_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.1
    )

    mesh = mesh.filter_smooth_taubin(number_of_iterations=15)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=.1
    )
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

    for i in range(9):
        if not i == 0 and i % 2 == 0:
            pcd = mesh.sample_points_uniformly(number_of_points=4096*i)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=.1
            )

    mesh = mesh.filter_smooth_simple(number_of_iterations=15)

    return mesh