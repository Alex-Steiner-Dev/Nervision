import open3d as o3d

gt_mesh = o3d.data.BunnyMesh()
pcd = gt_mesh.sample_points_poisson_disk(3000)
pcd.compute_vertex_normals()

radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])