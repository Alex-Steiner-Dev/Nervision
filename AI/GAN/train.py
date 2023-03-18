import torch
import torch.optim as optim


from model import Generator, Discriminator

from gradient_penalty import GradientPenalty
from data_benchmark import BenchmarkDataset
from stitchingloss import stitchloss

from arguments import Arguments

import time
import numpy as np

class GAN():
    def __init__(self, args):
        self.args = args
      
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, class_choice=args.class_choice)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        print("Training Dataset : {} prepared.".format(len(self.data)))

        self.G = Generator(num_points=2048).to(args.device)      
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")

    def run(self):        
        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}

        for epoch in range(epoch_log, self.args.epochs):
            for _iter, data in enumerate(self.dataLoader):
                point, _ = data
                point = point.to(self.args.device)
                start_time = time.time()
       
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()

                    z = torch.randn(self.args.batch_size, 1, 128).to(self.args.device)

                    with torch.no_grad():
                        fake_point = self.G(z)         
                        fake_point = (fake_point)

                    D_real, real_index = self.D(point)
                    D_realm = D_real.mean()
                    D_fake, _ = self.D(fake_point)
                    D_fakem = D_fake.mean()

                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                realvar = stitchloss(point, real_index)

                loss_log['D_loss'].append(d_loss.item())                  
                
                self.G.zero_grad()
                z = torch.randn(self.args.batch_size, 1, 128).to(self.args.device)
                fake_point = self.G(z)
                fake_point = (fake_point)
                G_fake, fake_index = self.D(fake_point)
                
                fakevar = stitchloss(fake_point,fake_index)
                G_fakem = G_fake.mean()
                
                varloss = torch.pow((fakevar-realvar),2)
                
                g_loss = -G_fakem + 0.05*varloss
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())
                 
                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))
           
            if epoch % 50 == 0:
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                }, str(epoch)+'.pt')

                print('Checkpoint is saved.')

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    model = GAN(args)
    model.run()