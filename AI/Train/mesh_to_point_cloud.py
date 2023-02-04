import trimesh
import numpy as np
import open3d as o3d

def mesh_to_point_cloud(mesh_location, file_location):
    mesh = trimesh.load(mesh_location)
    vertices = mesh.vertices
    np.savetxt(file_location + "/point_cloud.pc", vertices)