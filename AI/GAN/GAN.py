import keras
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, ReLU, LeakyReLU
from keras.layers.convolutional import Conv3D, Convolution3DTranspose

class Generator():
    def __init__(self):
        self.cube_len = 32
        self.bias = False
        self.z_dim = 200
        self.f_dim = 32
        padd = (1,1,1)

        self.layer1 = self.conv_layer(self.z_dim, self.f_dim*8, kernel_size=4, stride=2, padding=padd, bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim*8, self.f_dim*4, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*4, self.f_dim*2, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*2, self.f_dim, kernel_size=4, stride=2, padding=(1, 1, 1), bias=self.bias)
        
        self.layer5 = Sequential(
            Convolution3DTranspose(self.f_dim, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1), activation="sigmoid"),
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = Sequential(
            Convolution3DTranspose(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            BatchNormalization(output_dim),
            ReLU(0.2)
        )
        return layer

class Discriminator():
    def __init__(self):
        self.cube_len = 32
        self.leak_value = 0.2
        self.bias = False
        padd = (1,1,1)
        self.f_dim = 32

        self.layer1 = self.conv_layer(1, self.f_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer2 = self.conv_layer(self.f_dim, self.f_dim*2, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer3 = self.conv_layer(self.f_dim*2, self.f_dim*4, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)
        self.layer4 = self.conv_layer(self.f_dim*4, self.f_dim*8, kernel_size=4, stride=2, padding=(1,1,1), bias=self.bias)

        self.layer5 = Sequential(
            Conv3D(self.f_dim*8, 1, kernel_size=4, stride=2, bias=self.bias, padding=padd, activation="sigmoid")
        )

    def conv_layer(self, input_dim, output_dim, kernel_size=4, stride=2, padding=(1,1,1), bias=False):
        layer = Sequential(
            Conv3D(input_dim, output_dim, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            BatchNormalization(output_dim),
            LeakyReLU(self.leak_value, inplace=True)
        )

        return layer