import open3d as o3d
import numpy as np

def mesh_to_point_cloud(mesh_location, file_location):
    pcd = o3d.io.read_point_cloud(mesh_location)
    vertices = np.asarray(pcd.points)

    step = int(vertices.shape[0]/5000)
    downsampled_vertices = vertices[::step, :]

    np.savetxt(file_location + "/point_cloud.pc", downsampled_vertices)
