from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution3D, UpSampling3D
import numpy as np

class GAN:
    def __init__(self, box_size):
        self.box_size = box_size

        self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates images
        z = Input(shape=(100,))
        img = self.generator(z)

        # Only train the generator for the combined model
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 16 * 16 * 16, activation='relu', input_dim=100))
        model.add(Reshape((16, 16, 16, 128)))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Convolution3D(64, (5, 5, 5), activation='relu', padding='same'))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Convolution3D(1, (5, 5, 5), activation='sigmoid', padding='same'))

        noise = Input(shape=(100,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Convolution3D(32, (5, 5, 5), strides=(2, 2, 2), padding='same', input_shape=(self.box_size, self.box_size, self.box_size, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Convolution3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Convolution3D(128, (5, 5, 5), strides=(2, 2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=(self.box_size, self.box_size, self.box_size, 1))
        validity = model(img)

        return Model(img, validity)

    def train(self, x_train, epochs, batch_size=32):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, 100))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise, valid)

            print(f"{epoch+1} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")

        self.generator.save("generator.")