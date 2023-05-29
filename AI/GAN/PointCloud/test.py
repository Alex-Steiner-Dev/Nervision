import trimesh
import numpy as np
import json
import pyvista as pv
import open3d as o3d
from mesh_generation import *


f = open("../captions.json")
data = json.load(f)

def create_mesh(vertices, faces):
    # Convert vertices and faces to NumPy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Create an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    
    # Set the vertices and faces of the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Update the mesh to compute the normals and other properties
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh

def simplify_mesh(mesh, n):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(n)
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01)
    count = 0.01
    while len(simplified_mesh.vertices) > 2048:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.01 / count)
        count = count - 0.01

    mesh = create_mesh(simplified_mesh.vertices, simplified_mesh.triangles)

    o3d.visualization.draw_geometries([mesh])
    
    return simplified_mesh

a = 0
for i, itObject in enumerate(data):
    obj_path = "../dataset/" + itObject['id'] + ".obj"
    simplified_mesh = simplify_mesh(obj_path, 2048)
    if len(simplified_mesh.vertices) > a:
        a = len(simplified_mesh.vertices)

print(a)

f.close()