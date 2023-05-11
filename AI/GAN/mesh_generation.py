import open3d as o3d
import numpy as np
import pyvista as pv

def generate_mesh(points):
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.05)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

            
    mesh = mesh.filter_smooth_simple(number_of_iterations=1)

    return mesh