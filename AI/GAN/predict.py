import torch
import pyvista as pv
from model import Generator
import numpy as np
import open3d as o3d

Generator = Generator(num_points=2048).cuda()

model_path = "chair.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.randn(1, 1, 128).cuda()

with torch.no_grad():
    sample = Generator(z).cpu()

    points = sample.numpy().reshape(2048,3)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.1
    )
    mesh = mesh.filter_smooth_taubin(number_of_iterations=15)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.1
    )
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=5000)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=0.1
    )

    o3d.io.write_triangle_mesh("output_mesh.obj", mesh)

    o3d.visualization.draw_geometries([mesh])