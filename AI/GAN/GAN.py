from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras import backend as K
import keras
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

class VAE:
    def __init__(self, box_size, resolutions):
        self.box_size = box_size
        self.resolutions = resolutions

    def build_vae(self):
        # Encoder
        input_img = Input(shape=(self.box_size, self.box_size, self.box_size, 1))
        x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
        x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same')(x)
        encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

        # Latent space
        latent_dim = 200
        flat = Flatten()(encoded)
        z_mean = Dense(latent_dim)(flat)
        z_log_var = Dense(latent_dim)(flat)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Decoder
        decoder_input = Input(shape=(latent_dim,))
        x = Dense(8*8*8*120, activation='relu')(decoder_input)
        x = Reshape((8, 8, 8, 120))(x)
        x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        decoded = Convolution3D(1, (5, 5, 5), activation='sigmoid', padding='same')(x)

        # VAE model
        encoder = Model(input_img, z_mean)
        vae_decoder = Model(decoder_input, decoded)
        z_decoded = vae_decoder(z)
        vae = Model(input_img, z_decoded)

        vae.compile(optimizer='adadelta', loss='binary_crossentropy')

        return vae, encoder, vae_decoder
