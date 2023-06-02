import torch
import torch.nn as nn
import model
import open3d as o3d
import numpy as np
from text_to_vec import *

def train():
    autoencoder = model.Autoencoder().to('cuda')
    criterion = nn.MSELoss().to('cuda')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    #######################################################################################
    Generator = model.Generator().cuda()

    vertices_path = "../TrainedModels/vertices.pt" 
    checkpoint = torch.load(vertices_path)
    Generator.load_state_dict(checkpoint['G_state_dict'])

    z = torch.from_numpy(text_to_vec(process_text(correct_prompt("old tractor"))).astype(np.float64)).reshape(1,512, 1).repeat(1, 1, 1).cuda().float()

    mesh = o3d.io.read_triangle_mesh("dataset/tractor.obj")
    simplified_mesh = mesh.simplify_quadric_decimation(4096)

    if len(simplified_mesh.vertices) > 4096:
        simplified_mesh = simplified_mesh.simplify_vertex_clustering(.0005)

    vertices = np.array(simplified_mesh.vertices)
        
    expanded_array = np.zeros((4096, 3))
    expanded_array[:vertices.shape[0], :] = vertices
 
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