import numpy as np
import pyvista as pv
import trimesh

def point_cloud_to_mesh_obj(points):
    point_cloud = pv.PolyData(points)
    mesh = point_cloud.reconstruct_surface()
    mesh.save('mesh.stl')
