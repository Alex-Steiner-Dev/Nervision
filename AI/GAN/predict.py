import torch
from mesh_generation import generate_mesh
from model import Generator
import open3d as o3d

Generator = Generator(num_points=2048).cuda()

model_path = "chair.pt" 
checkpoint = torch.load(model_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.randn(1, 1, 128).cuda()

with torch.no_grad():
    sample = Generator(z).cpu()

    points = sample.numpy().reshape(2048,3)

    mesh = generate_mesh(points)

    o3d.io.write_triangle_mesh("generation.obj", mesh)

    o3d.visualization.draw_geometries([mesh])