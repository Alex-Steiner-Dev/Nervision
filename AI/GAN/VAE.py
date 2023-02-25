from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D

class VAE:
    def __init__(self, box_size, resolutions):
        self.box_size = box_size
        self.resolutions = resolutions

    def build_vae(self):
        # Encoder
        input_img = Input(shape=(self.box_size, self.box_size, self.box_size, 1))
        x = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)

        for i in self.resolutions:
            x = Convolution3D(i, (3, 3, 3), activation='relu', padding='same')(x)
            x = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Convolution3D(1024, (3, 3, 3), activation='relu', padding='same')(x)
        encoder = MaxPooling3D((2, 2, 2), padding='same')(x)

        #Decoder
        self.resolutions.reverse()

        x = Convolution3D(1024, (3, 3, 3), activation='relu', padding='same')(encoder)
        x = UpSampling3D((2, 2, 2))(x)

        for i in self.resolutions:
            x = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(x)
            x = UpSampling3D((2, 2, 2))(x)

        x = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(x)
        decoder = Convolution3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoder)

        autoencoder.compile(optimizer='adam', loss='mse')

        return (autoencoder, encoder, decoder)