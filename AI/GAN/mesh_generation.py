import open3d as o3d
import numpy as np
import trimesh

def generate_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    mesh = trimesh.Trimesh(np.asarray(pcd.points), np.asarray(pcd.normals))

    # Convert the trimesh object to an Open3D mesh object
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    # Clean the mesh
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_unreferenced_vertices()

    return o3d_mesh
    return mesh