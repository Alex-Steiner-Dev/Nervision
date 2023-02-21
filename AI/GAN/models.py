from keras.layers import Input, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv3D, Convolution3DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Generator():
    def __init__(self):
        self.latent_dim = 200
        self.kernel_size = (4,4,4)
        self.strides = (2, 2, 2)


    def build_generator(self):
        model = Sequential()
        model.add(Convolution3DTranspose(filters=512, kernel_size=self.kernel_size,
                  strides=(1, 1, 1), kernel_initializer='glorot_normal',
                  bias_initializer='zeros', padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3DTranspose(filters=256, kernel_size=self.kernel_size,
                      strides=self.strides, kernel_initializer='glorot_normal',
                      bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3DTranspose(filters=128, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3DTranspose(filters=64, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3DTranspose(filters=1, kernel_size=self.kernel_size,
                           strides=self.strides, kernel_initializer='glorot_normal',
                           bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

        noise = Input(shape=(1, 1, 1, self.latent_dim))
        image = model(noise)

        return Model(inputs=noise, outputs=image)

class Discriminator():
    def __init__(self):
        self.im_dim = 64
        self.latent_dim = 200
        self.kernel_size = (4,4,4)
        self.strides = (2, 2, 2)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv3D(filters=64, kernel_size=self.kernel_size,
                    strides=self.strides, kernel_initializer='glorot_normal',
                    bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=128, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=256, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=512, kernel_size=self.kernel_size,
                         strides=self.strides, kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv3D(filters=1, kernel_size=self.kernel_size,
                         strides=(1, 1, 1), kernel_initializer='glorot_normal',
                         bias_initializer='zeros', padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('sigmoid'))

        image = Input(shape=(self.im_dim, self.im_dim, self.im_dim, 1))
        validity = model(image)

        return Model(inputs=image, outputs=validity)
