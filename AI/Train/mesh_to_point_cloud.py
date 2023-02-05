import trimesh
import numpy as np

def mesh_to_point_cloud(mesh_location, file_location):
    mesh = trimesh.load(mesh_location)
    points = mesh.sample(count=5000)
    np.savetxt(file_location + "/point_cloud.pc", points)

    import open3d as o3d

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_location + "/point_cloud.pc", format='xyz')

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])