import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape
from keras import backend as K
from keras import losses
import text
import trimesh

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def create_vae(latent_dim, input_shape):

    # Encoder architecture
    x = Input(shape=input_shape)
    h = Dense(4096, activation='relu')(x)
    h = Dense(2048, activation='relu')(h)
    h = Dense(1024, activation='relu')(h)
    h = Dense(512, activation='relu')(h)
    h = Dense(256, activation='relu')(h)
    h = Dense(128, activation='relu')(h)
    h = Dense(64, activation='relu')(h)

    # Latent space
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder architecture
    decoder_h = Dense(3, activation='relu')
    decoder_reshape = Reshape((4096, 3))
    decoder_mean = Dense(4096*3, activation='sigmoid')
    h_decoded = decoder_h(z)
    h_decoded = decoder_reshape(h_decoded)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    vae = Model(x, x_decoded_mean)

    # Define the loss function
    def vae_loss(x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    # Encoder model
    encoder = Model(x, z_mean)

    return vae, encoder

# Load the 3D models and their corresponding text descriptions
x_train = np.array([trimesh.load_mesh("../Data/chair/train/chair_0001.off").sample(4096)])
y_train = text.word_embedding(["a chair"])

# Define the input shape and latent dimension
input_shape = (4096,3)
latent_dim = 4096

# Create the VAE model
vae, encoder = create_vae(latent_dim, input_shape)

# Train the model
batch_size =1
epochs = 100

vae.fit(x_train, x_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1)

def generate_3d_model(text_description, encoder):
    # Preprocess the text description
    text_vector = text.word_embedding(text_description)

    # Encode the text vector to the latent space
    z = encoder.predict(text_vector)

    return vae.predict(z)

generate_3d_model("a chair", encoder)