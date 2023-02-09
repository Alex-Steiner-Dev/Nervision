import numpy as np
import pyvista as pv

def point_cloud_to_mesh_obj(points):
    cloud = pv.PolyData(points)

    volume = cloud.delaunay_3d(alpha=2.)
    shell = volume.extract_geometry()
    shell.save('mesh.stl')

