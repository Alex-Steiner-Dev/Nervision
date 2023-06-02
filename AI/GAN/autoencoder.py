import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from predict import *

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(2048, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 128, 1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 512, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 2048, 1),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder().to('cuda')
criterion = nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

mesh = o3d.io.read_triangle_mesh("dataset/40f1be4ede6113a2e03aea0698586c31.obj")
simplified_mesh = mesh.simplify_quadric_decimation(2048)

if len(simplified_mesh.vertices) > 2048:
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

vertices = np.array(simplified_mesh.vertices)
    
expanded_array = np.zeros((2048, 3))
expanded_array[:vertices.shape[0], :] = vertices
                
point_cloud = np.array(expanded_array, dtype=np.float32)

num_epochs = 2000

target_data = torch.from_numpy(expanded_array)
input_data = torch.from_numpy(predict())

for epoch in range(num_epochs):
    input_data = input_data.float().to('cuda')
    target_data = target_data.float().to('cuda')

    outputs = autoencoder(input_data)
    loss = criterion(outputs, target_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

denoised_output = autoencoder(torch.from_numpy(predict()).float().to('cuda')).cpu().detach().numpy()

def create_mesh(vertices, faces):
    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = o3d.geometry.TriangleMesh()
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh

mesh = create_mesh(denoised_output, simplified_mesh.triangles)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(denoised_output)

o3d.visualization.draw_geometries([mesh])
