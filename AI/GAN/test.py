import trimesh
import numpy as np
import json
import pyvista as pv
import open3d as o3d
import pyvista as pv

f = open("captions.json")
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
    simplified_mesh = mesh.simplify_quadric_decimation(2048)

    if len(simplified_mesh.vertices) > 2048:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

    vertices = np.array(simplified_mesh.vertices)
    
    expanded_array = np.zeros((2048, 3))
    expanded_array[:vertices.shape[0], :] = vertices
                
    point_cloud = np.array(expanded_array, dtype=np.float32)

    print(point_cloud[0])

    mesh = create_mesh(point_cloud, simplified_mesh.triangles)

    o3d.visualization.draw_geometries([mesh])
    
    return simplified_mesh


obj_path = "dataset/40f1be4ede6113a2e03aea0698586c31.obj"
simplified_mesh = simplify_mesh(obj_path, 2048)

f.close()