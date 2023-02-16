import tensorflow as tf
import numpy as np
from dataset import parse_dataset

def generator(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(2048),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(4096, activation='tanh')
    ])
    return model

def discriminator(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(2048, input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def gan(generator, discriminator, input_shape):
    discriminator.trainable = False
    model = tf.keras.models.Sequential([
        generator,
        discriminator
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

data = parse_dataset()

input_shape = (2048,3)

generator = generator(input_shape)
discriminator = discriminator(input_shape)

gan = gan(generator, discriminator, input_shape)

for epoch in range(10000):
    noise = np.random.rand(2048, 3)
    print(noise.shape)
    #fake_point_clouds = generator.predict(noise)

    """
    real_point_clouds = data[0]

    discriminator.trainable = True
    discriminator.train_on_batch(real_point_clouds, np.ones((32, 1)))
    discriminator.train_on_batch(fake_point_clouds, np.zeros((32, 1)))

    discriminator.trainable = False
    gan.train_on_batch(noise, np.ones((32, 1)))

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {gan.evaluate(noise, np.ones((32, 1)))}")
    """