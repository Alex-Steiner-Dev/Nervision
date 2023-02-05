import trimesh
import numpy as np

def mesh_to_point_cloud(mesh_location, file_location):
    mesh = trimesh.load(mesh_location)
    points = mesh.sample(count=5000)
    np.savetxt(file_location + "/point_cloud.pc", points)