from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
import numpy as np
import trimesh

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

max_seq_length = 100
latent_dim = 32
mesh_dim = 5000

texts = {"a chair"}
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

input_text = Input(shape=(max_seq_length,), dtype='int32')
h = Dense(256, activation='relu')(input_text)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

chair_mesh = trimesh.load('../Data/chair/train/chair_0001.off')
points = chair_mesh.sample(5000)

points = (points - points.mean(axis=0)) / points.std(axis=0)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

decoder_input = Input(shape=(latent_dim,))
h = Dense(256, activation='relu')(decoder_input)
output_mesh = Dense(mesh_dim, activation='sigmoid')(h)


vae = Model(input_text, output_mesh)

encoder = Model(input_text, z_mean)
decoder = Model(decoder_input, output_mesh)

def vae_loss(input_mesh, output_mesh):
    mse_loss = K.mean(K.square(input_mesh - output_mesh), axis=-1)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return mse_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

vae.fit(points, epochs=1000, batch_size=1)

def generate_mesh(text):
    text_seq = tokenizer.texts_to_sequences([text])
    padded_text_seq = pad_sequences(text_seq, maxlen=max_seq_length, padding='post')
    z_mean, z_log_var, z = encoder.predict(padded_text_seq)
    generated_mesh = decoder.predict(z)
    return generated_mesh[0]

generate_mesh("a chair")