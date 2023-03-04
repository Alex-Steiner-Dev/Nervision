from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D

class VAE:
    def __init__(self, box_size):
        self.box_size = box_size

    def build_vae(self):
        # Encoder
        input_img = Input(shape=(self.box_size, self.box_size, self.box_size, 1))
        x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
        x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same')(x)
        encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

        # Decoder
        decoder_input = Input(shape=(self.box_size // 4, self.box_size // 4, self.box_size // 4, 120))
        x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(decoder_input)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)

        decoder = Model(decoder_input, decoded, name='decoder')

        # Autoencoder
        autoencoder = Model(input_img, decoder(encoded))

        # Compile
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder, decoder