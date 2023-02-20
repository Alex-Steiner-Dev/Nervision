from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from dataset import parse_dataset
import numpy as np

dataset = parse_dataset()

class LSGAN():
    def __init__(self):
        self.shape = (4096, 3)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        mesh = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(mesh)

        self.combined = Model(z, valid)
        
        self.combined.compile(loss='mse', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.shape), activation='tanh'))
        model.add(Reshape(self.shape))

        noise = Input(shape=(self.latent_dim,))
        mesh = model(noise)

        return Model(noise, mesh)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        
        img = Input(shape=self.shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            real = dataset[0][0]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_model = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_model, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        self.combined.save("model.h5")

if __name__ == '__main__':
    gan = LSGAN()
    gan.train(epochs=30000, batch_size=4096, sample_interval=200)