import torch
from torch import optim
from torch import nn
from utils import *

from model import net_G, net_D

import time
import numpy as np
import params
from tqdm import tqdm

def trainer():
    dsets_path = "../Data/VolumetricData/chair/30/train/"
    train_dsets = ShapeNetDataset(dsets_path)
    train_dset_loaders = torch.utils.data.DataLoader(train_dsets, batch_size=params.batch_size, shuffle=True, num_workers=1)
    
    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}

    D = net_D()
    G = net_G()

    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    D.to(params.device)
    G.to(params.device)

    criterion_D = nn.MSELoss()

    criterion_G = nn.L1Loss()

    for epoch in range(params.epochs):
        start = time.time()

        for phase in ['train']:
            if phase == 'train':
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                X = X.to(params.device)
                batch = X.size()[0]
                Z = generateZ(batch)

                d_real = D(X)

                fake = G(Z)
                d_fake = D(fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
 
                d_real_loss = criterion_D(d_real, real_labels)

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

                if d_total_acu < params.d_thresh:
                    D.zero_grad()
                    d_loss.backward()
                    D_solver.step()

                Z = generateZ(batch)

                fake = G(Z)
                d_fake = D(fake)

                adv_g_loss = criterion_D(d_fake, real_labels)
                recon_g_loss = criterion_G(fake, X)

                g_loss = adv_g_loss

                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                G_solver.step()

                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)

            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]

            end = time.time()
            epoch_time = end - start

            print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
            print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

            if (epoch + 1) % params.model_save_step == 0:
                torch.save(G.state_dict(), "models" + '/G.pth')
                torch.save(D.state_dict(), "models" + '/D.pth')

                samples = fake.cpu().data[:8].squeeze().numpy()

                SavePloat_Voxels(samples, "images", epoch)
