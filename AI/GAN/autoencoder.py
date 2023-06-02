import torch
import torch.nn as nn
from model import *
import open3d as o3d
import numpy as np
from text_to_vec import *

autoencoder = Autoencoder().to('cuda')
criterion = nn.MSELoss().to('cuda')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

#######################################################################################
Generator = Generator().cuda()
Autoencoder = Autoencoder().cuda()

vertices_path = "../TrainedModels/vertices.pt" 
checkpoint = torch.load(vertices_path)
Generator.load_state_dict(checkpoint['G_state_dict'])

z = torch.from_numpy(text_to_vec(process_text(correct_prompt("cocktail table that is tall and square and average size"))).astype(np.float64)).reshape(1,512, 1).repeat(1, 1, 1).cuda().float()

mesh = o3d.io.read_triangle_mesh("dataset/40f1be4ede6113a2e03aea0698586c31.obj")
simplified_mesh = mesh.simplify_quadric_decimation(2048)

if len(simplified_mesh.vertices) > 2048:
    simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

vertices = np.array(simplified_mesh.vertices)
    
expanded_array = np.zeros((2048, 3))
expanded_array[:vertices.shape[0], :] = vertices
                
point_cloud = np.array(expanded_array, dtype=np.float32)

with torch.no_grad():
    sample = Generator(z).cpu()
    vertices = sample.numpy()[0]

    vertices = np.array(vertices, dtype=np.float32)
#######################################################################################

num_epochs = 2000

target_data = torch.from_numpy(expanded_array)
input_data = torch.from_numpy(vertices)

for epoch in range(num_epochs):
    input_data = input_data.float().to('cuda')
    target_data = target_data.float().to('cuda')

    outputs = autoencoder(input_data)
    loss = criterion(outputs, target_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save({'autoencoder': autoencoder.state_dict()}, '../TrainedModels/autoencoder.pt')