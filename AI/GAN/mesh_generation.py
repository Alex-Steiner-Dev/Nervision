import open3d as o3d

def generate_mesh(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.1
    )

    mesh = mesh.filter_smooth_simple(number_of_iterations=1)

    return mesh