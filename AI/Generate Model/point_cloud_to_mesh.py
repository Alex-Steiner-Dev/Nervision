import numpy as np
from scipy.spatial import Delaunay
import trimesh

def point_cloud_to_mesh_obj(file_name):
    points = np.random.random((100, 3))
    tri = Delaunay(points)
    faces = tri.simplices
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    mesh.export(file_name)

file_name = "mesh.obj"
point_cloud_to_mesh_obj(file_name)