# %%
import os
from plyfile import PlyData
import time

import torch
from torch import nn, optim
from torch import cuda, device, utils
from torch.autograd import Variable

from torchsummary import summary
from torchvision import datasets, transforms

import numpy as np
import pyvista as pv

print(cuda.is_available())
device = device('cuda:0')
print(device)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pcl(points, ax, nogrid):
    ax.patch.set_alpha(0)
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.view_init(elev=10., azim=240.)
    if nogrid: ax.grid(False)
    ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,1], cmap='plasma', s=10)

def plot_pcls(pcls, nogrid=False):
    fig = plt.figure(figsize=(16, 8))
    for i, pcl in enumerate(pcls):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        plot_pcl(pcl, ax, nogrid)

# %%
class Decoder(nn.Module):
    def __init__(self, nc, nf, nz):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, nf),
            nn.ReLU(True),
            nn.Linear(nf, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048*3),
        )
    
    def forward(self, input):
        return self.net(input).view(input.size(0), 3, -1)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

# %%
class Discriminator(nn.Module):
    def __init__(self, nc, nf, nz):
        super(Discriminator, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(nc, nf, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv1d(nf, 128, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv1d(128, 256, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv1d(256, 256, 1, 1),
            nn.LeakyReLU(0.1),
            
            nn.Conv1d(256, 512, 1, 1),
            nn.LeakyReLU(0.1),

#             nn.MaxPool1d(2048),
        )
        self.dec = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, nf),
            nn.ReLU(True),
            nn.Linear(nf, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        z = self.enc(input)
        z, _ = torch.max(z, dim=-1)
        prob = self.dec(z)
        return prob
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

# %%
nz = 128
nf = 64
nc = 3;
mu = 0.
sigma = 0.2
batch_size = 8

dec = Decoder(nc=nc, nf=nf, nz=nz).to(device)
dec.apply(dec.weights_init)
z = torch.FloatTensor(batch_size, nz).normal_(0, 0.2).to(device)
print(z.shape)
print(dec(z).shape)
summary(dec, (1, nz))

dis = Discriminator(nc=nc, nf=nf, nz=nz).to(device)
dis.apply(dis.weights_init)
y = Variable(torch.ones((batch_size, nc, 2048)).to(device))
print(y.shape)
prob = dis(y)
print(prob.shape)
summary(dis, (nc, 2048))

# %%

def ShapeNetLoader(input):
    arr = np.fromfile(input, dtype=np.float32).reshape(-1, 3)
    return torch.from_numpy(arr)

class PCLDataset(utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.listdir = os.listdir(self.root)

        data_size = len(self.listdir)
        self.listdir = self.listdir[0:int(data_size)]
        
        print ('data_size =', len(self.listdir)) 

    def __getitem__(self, index):
        ply_data = PlyData.read(self.listdir[index])
        points = ply_data['vertex']
        points = np.vstack([points['x'], points['y'], points['z']]).T
        ret_val = [points]

        return torch.FloatTensor(ret_val)

    def __len__(self):
        return len(self.listdir)

    
shapenet_path = 'shape_net_core_uniform_samples_2048/03001627/'

batch_size = 50
dataset = PCLDataset(shapenet_path)
dataloader = utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print('Batch size: ', batch_size)

print('Number of batches:', len(dataloader))


# %%
idx, (dataA) = next(enumerate(dataloader))
x = Variable(dataA).to(device)
print(x.shape)
plot_pcls([x.cpu().detach().numpy()[0].transpose(1,0)])

# %%
lrg = 1e-4
lrd = 1e-4
b1 = 0.5
b2 = 0.999
wd = 1e-3

optimizer_gen = optim.Adam(params=dec.parameters(), lr=lrg, betas=(b1, b2))
optimizer_dis = optim.Adam(params=dis.parameters(), lr=lrd, betas=(b1, b2))

criterion_adv = nn.BCELoss().to(device) 

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()

    def __call__(self, synt):
        return torch.mean(-torch.log(synt)) 

criterion_gen = GenLoss().to(device)

class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()

    def __call__(self, real, synt):
        return torch.mean(-torch.log(real) - torch.log(1 - synt))

criterion_dis = DisLoss().to(device)

# %%
epochs = 200

gen_losses = []
dis_losses = []

print_iter = 10
print_iter = len(dataloader) / 2#20
for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    
    for idx, (dataA) in enumerate(dataloader):
        print(idx, end='\r')
        start = time.time()
        batch_size = dataA.shape[0]
#         if idx == 1: break
        
        x = Variable(dataA).to(device)
        z = torch.FloatTensor(batch_size, nz).normal_(0., 0.2).to(device)
        rec = dec(z)
        
        prob_x = dis(x)
        prob_rec = dis(rec)
        prob_real = Variable(torch.ones((batch_size, 1)).to(device))
        prob_fake = Variable(torch.zeros((batch_size, 1)).to(device))
        gen_loss = criterion_adv(prob_rec, prob_real)
#         gen_loss = criterion_gen(prob_rec)

        optimizer_gen.zero_grad()
        gen_loss.backward(retain_graph=True)
        optimizer_gen.step()
        gen_losses.append(gen_loss.item())
        
        optimizer_dis.zero_grad()
        dis_real_loss = criterion_adv(prob_x, prob_real)
        dis_rec_loss = criterion_adv(prob_rec, prob_fake)
        dis_loss = dis_real_loss + dis_rec_loss
#         dis_loss = criterion_dis(prob_x, prob_rec)
        
        dis_loss.backward()
        optimizer_dis.step()
        dis_losses.append(dis_loss.item())
        
        if not (idx % print_iter):
            print('[%d/%d;%d/%d]: gen_loss: %.3f,'
                  ' dis_loss: %.3f'
              % (idx, len(dataloader), (epoch), epochs, 
                 torch.mean(torch.FloatTensor(gen_losses)),
                 torch.mean(torch.FloatTensor(dis_losses))))
    print('Time: ', time.time() - epoch_start)

# %%
def running_mean(data_set, periods=10):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

rm_gen = running_mean(np.asarray(gen_losses))
rm_dis = running_mean(np.asarray(dis_losses))
plt.plot(range(0, len(rm_gen)), rm_gen, 'r-', label='Generator')
plt.plot(range(0, len(rm_dis)), rm_dis, 'b-', label='Discriminator')
plt.gca().patch.set_alpha(0)
plt.gcf().set_size_inches(8,4)
plt.title('Total Losses')
plt.legend()
plt.show();

# %%
idx, (dataA) = next(enumerate(dataloader))
x = Variable(dataA).to(device)
# rec = decS(encS(x))
# visu_pcl(x.cpu().detach().numpy()[0].transpose(1,0), rec.cpu().detach().numpy()[0].transpose(1,0))
# visu_pcl(x.cpu().detach().numpy()[1].transpose(1,0), rec.cpu().detach().numpy()[1].transpose(1,0))
z = torch.Tensor(4, nz).normal_(0., 0.2).to(device)
rec = dec(z)
# print(rec)
print(rec.shape)
plot_pcls([x.cpu().detach().numpy()[0].transpose(1,0), rec.cpu().detach().numpy()[0].transpose(1,0)])
plot_pcls([x.cpu().detach().numpy()[1].transpose(1,0), rec.cpu().detach().numpy()[1].transpose(1,0)])
plot_pcls([x.cpu().detach().numpy()[2].transpose(1,0), rec.cpu().detach().numpy()[2].transpose(1,0)])
plot_pcls([x.cpu().detach().numpy()[3].transpose(1,0), rec.cpu().detach().numpy()[3].transpose(1,0)])


