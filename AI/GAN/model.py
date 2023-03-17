from keras.layers import Conv1D, LeakyReLU, MaxPooling1D, Permute, Flatten, Dense, Concatenate
from keras.models import Model, Sequential
import itertools
import numpy as np

class CodeEnhancement(Model):
    def __init__(self, num_points=2048):
        super(CodeEnhancement, self).__init__()
        self.conv1 = Conv1D(128, kernel_size=1, use_bias=False)
        self.conv2 = Conv1D(128, kernel_size=1, use_bias=False)
        self.conv3 = Conv1D(256, kernel_size=1, use_bias=False)
        self.conv4 = Conv1D(256, kernel_size=1, use_bias=False)
        self.conv5 = Conv1D(512, kernel_size=1, use_bias=False)
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.leaky_relu(self.conv1(Permute((2, 1))(inputs)))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))

        return Permute((2, 1))(x)

class WarpingGAN(Model):
    def __init__(self, num_points=2048, m=128, dimofgrid=3):
        super(WarpingGAN, self).__init__()
        self.n = num_points  
        self.numgrid = int(2048 / m)
        self.dimofgrid = dimofgrid
        self.m = m  
        self.meshgrid = [[-0.2, 0.1, 4], [-0.2, 0.1, 4], [-0.2, 0.1, 8]]
        self.codegenerator = CodeEnhancement(num_points=num_points)
        self.mlp1 = Sequential([
            Conv1D(filters=256, kernel_size=1),
            LeakyReLU(alpha=0.2),
            Conv1D(filters=64, kernel_size=1),
            LeakyReLU(alpha=0.2),
            Conv1D(filters=3, kernel_size=1),
        ])

        self.mlp2 = Sequential([
            Conv1D(filters=256, kernel_size=1),
            LeakyReLU(alpha=0.2),
            Conv1D(filters=64, kernel_size=1),
            LeakyReLU(alpha=0.2),
            Conv1D(filters=3, kernel_size=1),
        ])

    def build_grid(self, batch_size):
        
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        z = np.linspace(*self.meshgrid[2])
        grid = np.array(list(itertools.product(x, y, z)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        
        return grid

    def forward(self, input):
        input = self.codegenerator(input)
        input = input.transpose(1, 2).repeat(1, 1, self.m)  
        
        batch_size = input.size(0)
        splitinput = input.view(batch_size, self.numgrid, int(512/self.numgrid), self.m)
        globalfeature = input.view(batch_size, 1, 512, self.m).repeat(1,self.numgrid,1,1)
        gridlist = []

        for i in range(self.numgrid):
            gridlist.append(self.build_grid(input.shape[0]).transpose(1, 2).view(batch_size, 1, self.dimofgrid, self.m))
        
        grid = gridlist[0]
        for i in range(1,self.numgrid): 
            grid = Concatenate(axis=1)(grid,gridlist[i])
        concate1 = Concatenate(dim=2)(splitinput, globalfeature, grid)
        concate1 = concate1.transpose(1,2).reshape(batch_size, int(512/self.numgrid)+self.dimofgrid+512, 2048)  
        after_folding1 = self.mlp1(concate1)  
        concate2 = Concatenate(dim=2)(splitinput, globalfeature, after_folding1.reshape(batch_size,3,self.numgrid,self.m).transpose(1,2))
        concate2 = concate2.transpose(1,2).reshape(batch_size, int(512/self.numgrid)+3+512, 2048)  
        after_folding2 = self.mlp2(concate2)  
        return after_folding2.transpose(1, 2)   


class Discriminator(Model):
    def __init__(self, batch_size, features, num_points=2048):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        self.numpoints = num_points
        self.fc_layers = Sequential()
        for inx in range(self.layer_num):
            self.fc_layers.add(Conv1D(features[inx+1], kernel_size=1, strides=1))

        self.leaky_relu = LeakyReLU(alpha=0.2)
        self.final_layer = Sequential(Dense(features[-1]),
                                              Dense(features[-2]),
                                              Dense(features[-2]),
                                              Dense(1))
        self.maxpool = MaxPooling1D(pool_size=self.numpoints)

    def call(self, f):
        feat = Permute((2, 1))(f)

        for inx in range(self.layer_num):
            feat = self.fc_layers.layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = self.maxpool(feat)
        out = Flatten()(out)
        out = self.final_layer(out)

        return out