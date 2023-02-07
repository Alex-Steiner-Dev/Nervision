import numpy as np
from scipy.spatial import Delaunay
import trimesh

def point_cloud_to_mesh_obj(points, file_name):
    mesh = trimesh.PointCloud(points)
    mesh = mesh.polygon_mesh(alpha=1.0)
    mesh.export("mesh.ply")

