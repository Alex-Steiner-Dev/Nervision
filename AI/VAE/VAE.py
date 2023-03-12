from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D

class VAE:
    def __init__(self, box_size):
        self.box_size = box_size

    def build_vae(self):
        input_img = Input(shape=(self.box_size, self.box_size, self.box_size, 1))

        x = Convolution3D(32, (5, 5, 5), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(64, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(128, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(256, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(512, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(1024, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(2048, (5, 5, 5), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Convolution3D(4096, (5, 5, 5), activation='relu', padding='same')(x)
        encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

        x = Convolution3D(2048, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(1024, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(512, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(256, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(128, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(64, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(32, (5, 5, 5), activation='relu', padding='same')(encoded)
        x = UpSampling3D((2, 2, 2))(x)

        decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return (autoencoder)