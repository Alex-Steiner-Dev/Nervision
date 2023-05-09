import torch
from model import Generator
import open3d as o3d
from text_to_vec import *

import numpy as np

Generator = Generator().cuda()

model_path = "../TrainedModels/model.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("straight chair that is average size and tall and regular width"))).astype(np.float64)).reshape(1,512, 1).repeat(16, 1, 1).cuda().float()

with torch.no_grad():
    sample = Generator(z).cpu()

    vertices = sample.numpy()[0]

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.05)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()


    o3d.visualization.draw_geometries([mesh, point_cloud])