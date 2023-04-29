import open3d as o3d
import numpy as np
import pyvista as pv

def generate_mesh(point_cloud):
    point_cloud = pv.PolyData(point_cloud)
    surface_mesh = point_cloud.delaunay_2d()

    mesh = surface_mesh.smooth(n_iter=100, relaxation_factor=0.1)

    return mesh