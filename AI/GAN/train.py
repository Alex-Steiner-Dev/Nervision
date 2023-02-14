import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from lsgan import build_generator, build_discriminator, build_train_step

epoch = 50
steps = 1000
image_size = (32, 32, 3)
noise_size = (2, 2, 32)
batch_size = 16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_of_data = x_train.shape[0]

generator = build_generator(noise_size)
discriminator = build_discriminator(image_size)
train_step = build_train_step(generator, discriminator)

for e in range(epoch):
    for s in range(steps):

        real_images = x_train[np.random.permutation(num_of_data)[:batch_size]]
        noise = np.random.normal(0.0, 1.0, (batch_size,) + noise_size)

        d_loss, g_loss = train_step(real_images, noise)
        print ("[{0}/{1}] [{2}/{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(e, epoch, s, steps, d_loss, g_loss))

    image = generator.predict(np.random.normal(size=(10 * 10,) + noise_size))
