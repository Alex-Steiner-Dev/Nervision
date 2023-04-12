from matplotlib import pyplot as plt
import torch
import torch.optim as optim
from gradient_penalty import GradientPenalty

from model import Generator, Discriminator

from dataset import LoadDataset
from arguments import Arguments

import time

class GAN():
    def __init__(self, args):
        self.args = args
      
        self.data = LoadDataset(data_dir=args.dataset_path)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size)
        print("Training Dataset : {} prepared.".format(len(self.data)))

        self.G = Generator().to(args.device)      
        self.D = Discriminator().to(args.device)             

        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(10, gamma=1, device=args.device)

        print("Network prepared.")

    def run(self):      
        G_losses = []
        D_losses = []
         
        for epoch in range(args.epochs):
            for _iter, data in enumerate(self.dataLoader):
                point, label = data
                point = point.to(self.args.device)
                z = label.to(self.args.device)
                z = torch.reshape(z, (self.args.batch_size, 1, 768)).to(self.args.device)

                start_time = time.time()

                for d_iter in range(5):
                    self.D.zero_grad()
                    
                    with torch.no_grad():
                        fake_point = self.G(z).reshape(1,2048,3)       
                        
                    D_real = self.D(point)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_point)
                    D_fakem = D_fake.mean()
                    
                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()

                self.G.zero_grad()
            
                fake_point = self.G(z).reshape(1,2048,3)
                G_fake = self.D(fake_point)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem
                g_loss.backward()
                self.optimizerG.step()

                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))

            if epoch % 500 == 0:
                torch.save({'G_state_dict': self.G.state_dict()}, str(epoch)+'.pt')
                print('Checkpoint is saved.')

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("graph.png")
        plt.show()

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+ str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    model = GAN(args)
    model.run()