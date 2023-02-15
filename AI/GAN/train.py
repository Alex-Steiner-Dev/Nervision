import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from lsgan import build_generator, build_discriminator, build_train_step
from dataset import parse_dataset

epoch = 50
steps = 1000
model_size = (2048, 3)
noise_size = (2048,3)
batch_size = 16

x_train = parse_dataset()

generator = build_generator(noise_size)
discriminator = build_discriminator(model_size)
train_step = build_train_step(generator, discriminator)

generator.summary()
discriminator.summary()

for e in range(epoch):
    for s in range(steps):
        real_models = x_train[0]
        noise = np.random.rand(1, 2048, 3)
        d_loss, g_loss = train_step(real_models, noise)
        print ("[{0}/{1}] [{2}/{3}] d_loss: {4:.4}, g_loss: {5:.4}".format(e, epoch, s, steps, d_loss, g_loss))

    model = generator.predict(np.random.rand(1, 2048, 3))
