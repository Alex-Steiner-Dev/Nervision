from models import Generator, Discriminator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from dataset import parse_dataset
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def train():
    num_epochs = 10000
    batch_size = 1

    dis_optim = Adam(learning_rate=1e-5, beta_1=0.5)
    gen_optim = Adam(learning_rate=0.0025, beta_1=0.5)

    discriminator = Discriminator().build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optim)

    generator = Generator().build_generator()
    z = Input(shape=(1, 1, 1, 200))
    img = generator(z)

    discriminator.trainable = True
    validity = discriminator(img)

    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=gen_optim)

    dl, gl = [],[]

    real = parse_dataset()[0]
    real = real.reshape(1, 64, 64, 64)

    for epoch in range(num_epochs):
        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, 200])
        fake = generator.predict(z)

        lab_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
        lab_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))

        d_loss_real = discriminator.train_on_batch(real, lab_real)
        d_loss_fake = discriminator.train_on_batch(fake, lab_fake)

        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, 200])
        g_loss = combined.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))

        dl.append(d_loss)
        gl.append(g_loss)
        avg_d_loss = round(sum(dl)/len(dl), 4)
        avg_g_loss = round(sum(gl)/len(gl), 4)

        print('Training epoch {}/{}, d_loss_real/avg: {}/{}, g_loss/avg: {}/{}'.format(epoch+1, num_epochs, round(d_loss, 4), avg_d_loss, round(g_loss, 4), avg_g_loss))
 
    generator.save_weights('Models/generator_epoch_' + str(epoch+1) + ".h5", True)
    discriminator.save_weights('Models/discriminator_epoch_' + str(epoch+1) + ".h5", True)

train()