import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the input dimensions
input_shape = (64, 64, 3)
latent_dim = 100
num_classes = 3  # Number of chair types

# Define the encoder network
encoder_inputs = keras.Input(shape=input_shape)
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")

# Define the decoder network
latent_inputs = keras.Input(shape=(latent_dim + num_classes,))
x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Define the VAE model
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x):
        z_mean, z_log_var = self.encoder(x)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x, y):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        z_cond = tf.concat([z, y], axis=1)
        reconstructed = self.decode(z_cond)
        return reconstructed
    
# Define the conditioning input layer
condition_inputs = keras.Input(shape=(num_classes,))
condition_outputs = layers.Dense(latent_dim, activation='relu')(condition_inputs)

# Combine the encoder and conditioning input layers
combined_inputs = layers.concatenate([encoder_inputs, condition_outputs])

# Create an instance of the CVAE model
cvae = CVAE(encoder, decoder)
cvae.compile(optimizer=keras.optimizers.Adam())

# Train the model
cvae.fit([x_train, y_train], x_train, epochs=epochs, batch_size=batch_size)

# Generate a new chair image
condition = [0, 1, 0]  # Conditioning input for a modern chair
z = tf.random.normal((1, latent_dim))
z_cond = tf.concat([z, condition], axis=1)
generated_chair = cvae.decode(z_cond)
