import open3d as o3d
import numpy as np
import pyvista as pv


def point_cloud_to_mesh_obj(points):
    cloud = pv.PolyData(points)
    cloud.plot()

    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()
    shell.plot()
