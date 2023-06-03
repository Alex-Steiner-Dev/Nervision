import torch
import torch.nn as nn
import model
import open3d as o3d
import numpy as np
from text_to_vec import *
from dataset import *

def train():
    data = LoadAutoEncoder()
    dataLoader = torch.utils.data.DataLoader(data, batch_size=13, shuffle=True)  

    print("Training Dataset : {} prepared.".format(len(data)))

    autoencoder = model.Autoencoder().to('cuda')
    criterion = nn.MSELoss().to('cuda')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 1000

    print("Network prepared.")

    for epoch in range(num_epochs):
        for _iter, data in enumerate(dataLoader):
            input_data, target_data = data

            input_data = input_data.float().to('cuda')
            target_data = target_data.float().to('cuda')

            outputs = autoencoder(input_data)
            loss = criterion(outputs, target_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    torch.save({'autoencoder': autoencoder.state_dict()}, '../TrainedModels/autoencoder.pt')