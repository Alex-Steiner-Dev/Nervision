import numpy as np
from scipy.spatial import Delaunay
import trimesh

def point_cloud_to_mesh_obj(points, file_name):
    tri = Delaunay(points)
    faces = tri.simplices
    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    mesh.export(file_name)

