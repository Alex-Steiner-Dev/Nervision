import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from dataset import parse_dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Define the generator network
def generator_model():
    input_noise = layers.Input(shape=(100,))
    x = layers.Dense(256)(input_noise)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(4096*3, activation='tanh')(x)
    output = layers.Reshape((4096, 3))(x)
    return models.Model(input_noise, output)

# Define the discriminator network
def discriminator_model():
    input_cloud = layers.Input(shape=(4096, 3))
    x = layers.Flatten()(input_cloud)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_cloud, output)

# Define the GAN model that combines the generator and discriminator networks
def gan_model(generator, discriminator):
    discriminator.trainable = False
    input_noise = layers.Input(shape=(100,))
    generated_cloud = generator(input_noise)
    validity = discriminator(generated_cloud)
    return models.Model(input_noise, validity)

# Define the loss function that combines adversarial loss and reconstruction loss
def combined_loss(y_true, y_pred, generated_cloud, real_cloud):
    adversarial_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(generated_cloud, real_cloud)
    return adversarial_loss + 0.1*reconstruction_loss

# Load and preprocess the input mesh
input_cloud = parse_dataset()[0]
input_cloud = input_cloud.reshape(1,4096, 3)

# Define the generator, discriminator, and GAN models
generator = generator_model()
discriminator = discriminator_model()
gan = gan_model(generator, discriminator)

# Compile the discriminator model
discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Compile the GAN model
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Train the GAN
epochs = 2000
batch_size = 1
num_batches = 1

for epoch in range(epochs):
    for batch_index in range(num_batches):
        # Train the discriminator on a batch of real and generated point clouds
        real_cloud_batch = input_cloud[batch_index*batch_size:(batch_index+1)*batch_size]
        noise_batch = np.random.normal(0, 1, (batch_size, 100))
        generated_cloud_batch = generator.predict(noise_batch)
        y_real = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_cloud_batch, y_real)
        discriminator_loss_fake = discriminator.train_on_batch(generated_cloud_batch, y_fake)
        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
        
        # Train the generator to fool the discriminator
        noise_batch = np.random.normal(0, 1, (batch_size, 100))
        y_real = np.ones((batch_size, 1))
        generator_loss = gan.train_on_batch(noise_batch, y_real)

        if batch_index % 100 == 0:
            print('Epoch: %d, Batch: %d/%d, Discriminator Loss: %f, Generator Loss: %f' % 
              (epoch, batch_index, num_batches, discriminator_loss, generator_loss))
            
generator.save("gan.h5")
