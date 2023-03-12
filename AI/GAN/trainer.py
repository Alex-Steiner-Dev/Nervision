import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
from dataset import *
import matplotlib.pyplot as plt

n_epochs = 1000
batch_size = 1
lr = 0.0002
b1 = 0.5
b2 = 0.99
latent_dim = 100
img_size = 32
sample_interval = 400

class Generator( nn.Module ):
    def __init__( self, d=64 ):
        super( Generator, self ).__init__()
        self.deconv1 = nn.ConvTranspose3d( latent_dim, d * 8, 4, 1, 1 )
        self.deconv1_bn = nn.BatchNorm3d( d * 8 )
        self.deconv2 = nn.ConvTranspose3d( d * 8, d * 4, 4, 2, 1 )
        self.deconv2_bn = nn.BatchNorm3d( d * 4 )
        self.deconv3 = nn.ConvTranspose3d( d * 4, d * 2, 4, 2, 1 )
        self.deconv3_bn = nn.BatchNorm3d( d * 2 )
        self.deconv4 = nn.ConvTranspose3d( d * 2, d, 4, 2, 1 )
        self.deconv4_bn = nn.BatchNorm3d( d )
        self.deconv5 = nn.ConvTranspose3d( d, 1, 4, 2, 1 )


    
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    
    def forward( self, input ):
        
        x = input.view( -1, 100, 1, 1,1 )
        x = F.relu( self.deconv1_bn( self.deconv1( x ) ) )
        x = F.relu( self.deconv2_bn( self.deconv2( x ) ) )
        x = F.relu( self.deconv3_bn( self.deconv3( x ) ) )
        x = F.relu( self.deconv4_bn( self.deconv4( x ) ) )
        x = F.tanh( self.deconv5( x ) )
        return x

class Discriminator( nn.Module ):
    
    def __init__( self, d=64 ):
        super( Discriminator, self ).__init__()
        self.conv1 = nn.Conv3d( 1, d, 4, 2, 1 )
        self.conv2 = nn.Conv3d( d, d * 2, 4, 2, 1 )
        self.conv2_bn = nn.BatchNorm3d( d * 2 )
        self.conv3 = nn.Conv3d( d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d( d * 4 )
        self.conv4 = nn.Conv3d( d * 4, d * 8, 4, 2, 1 )
        self.conv4_bn = nn.BatchNorm3d( d * 8 )
        self.conv5 = nn.Conv3d( d * 8, 1, 4, 1, 1)

    
    def weight_init( self, mean, std ):
        for m in self._modules:
            normal_init( self._modules[ m ], mean, std )

    
    def forward( self, input ):
        x = F.leaky_relu( self.conv1( input ), 0.2 )
        x = F.leaky_relu( self.conv2_bn( self.conv2( x ) ), 0.2 )
        x = F.leaky_relu( self.conv3_bn( self.conv3( x ) ), 0.2 )
        x = F.leaky_relu( self.conv4_bn( self.conv4( x ) ), 0.2 )
        x = F.sigmoid( self.conv5( x ) )
        return x

def normal_init( m, mean, std ):
    if isinstance( m, nn.ConvTranspose2d ) or isinstance( m, nn.Conv2d ):
        m.weight.data.normal_( mean, std )
        m.bias.data.zero_()

def save(gen_voxels, num):
    voxel_data = gen_voxels[0,0].cpu().detach().numpy()
    
    voxel_data = voxel_data * (1.0 / voxel_data.max())
    
    voxel_data = voxel_data > 0.2

    
    fig = plt.figure()
    ax = fig.gca(projection='3d')                
    ax.voxels(voxel_data,  edgecolors='k')
    plt.savefig('images/voxels{}.png'.format(num))

def main(cuda):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    adversarial_loss = torch.nn.BCELoss()
    
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    
    generator.weight_init( mean=0.0, std=0.02 )
    discriminator.weight_init( mean=0.0, std=0.02 )
    


    mnist_3d_dataset = LoadDataset()
    train_loader = torch.utils.data.DataLoader( mnist_3d_dataset,
                                              batch_size=batch_size,
                                              shuffle=False )
    
    optimizer_G = torch.optim.Adam( generator.parameters(),
                                    lr=lr,
                                    betas=( b1, b2 ) )
    optimizer_D = torch.optim.Adam( discriminator.parameters(),
                                    lr=lr,
                                    betas=( b1, b2 ) )
    
    
    os.makedirs( 'images', exist_ok=True )
    os.makedirs( 'models', exist_ok=True )

    for epoch in range( n_epochs ):
        if ( epoch + 1 ) == 11:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )
        if ( epoch + 1 ) == 16:
            optimizer_G.param_groups[ 0 ][ 'lr' ] /= 10
            optimizer_D.param_groups[ 0 ][ 'lr' ] /= 10
            print( 'learning rate change!' )

        for i, voxels in enumerate( train_loader ):
            
            valid = Variable(Tensor(voxels.shape[0], 1, 1, 1, 1).fill_(1.0), requires_grad=False)

            fake = Variable( Tensor( voxels.shape[ 0 ], 1 ).fill_( 0.0 ),
                             requires_grad=False )
            
            real_voxels = Variable( voxels.type( Tensor ) )
     
            optimizer_G.zero_grad()
            
            z = Variable( Tensor( np.random.normal( 0, 1, ( voxels.shape[ 0 ],
                                                            latent_dim ) ) ) )
            
            gen_voxels = generator( z )
            
            g_loss = adversarial_loss( discriminator( gen_voxels ), valid )
            g_loss.backward()
            optimizer_G.step()
 
            optimizer_D.zero_grad()
            
            label_real = discriminator( real_voxels )
            label_gen = discriminator( gen_voxels.detach() )
            real_loss = adversarial_loss( label_real, valid )
            fake_loss = adversarial_loss( label_gen, fake )
            d_loss = ( real_loss + fake_loss ) / 2
            real_acc = ( label_real > 0.5 ).float().sum() / real_voxels.shape[ 0 ]
            gen_acc = ( label_gen < 0.5 ).float().sum() / gen_voxels.shape[ 0 ]
            d_acc = ( real_acc + gen_acc ) / 2
            d_loss.backward()
            optimizer_D.step()
        
            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % \
                    ( epoch,
                      n_epochs,
                      i,
                      len(train_loader),
                      d_loss.item(),
                      d_acc * 100,
                      g_loss.item() ) )
            
            batches_done = epoch * len( train_loader ) + i

            if batches_done % sample_interval == 0:
                save(gen_voxels, batches_done)
                
                torch.save( generator, 'models/gen_%d.pt' % batches_done )
                torch.save( discriminator, 'models/dis_%d.pt' % batches_done )

if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    main(cuda)