import pyvista as pv
import point_cloud_utils as pcu
import numpy as np

def generate_mesh(points):
    mesh = pv.PolyData(points).delaunay_3d(0.045).extract_geometry()

    return mesh