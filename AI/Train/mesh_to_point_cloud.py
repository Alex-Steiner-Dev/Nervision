import numpy as np
import trimesh

def mesh_to_point_cloud(mesh_location, file_location):
    mesh = trimesh.load(mesh_location)
    vertices = mesh.vertices

    step = int(vertices.shape[0]/10000)
    downsampled_vertices = vertices[::step, :]

    np.savetxt(file_location + "/point_cloud.pc", downsampled_vertices)
