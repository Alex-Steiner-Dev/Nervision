import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

class PointGAN:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()

        # Compile the discriminator
        self.discriminator.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=["accuracy"])

        # The generator takes noise as input and generates point clouds
        z = layers.Input(shape=(self.latent_dim,))
        point_cloud = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated point clouds as input and determines validity
        validity = self.discriminator(point_cloud)

        # The combined model (generator and discriminator)
        self.combined = keras.models.Model(z, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(0.0002, 0.5))

    def build_generator(self):
        model = keras.models.Sequential()
        model.add(layers.Dense(256, input_dim=self.latent_dim, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1024, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(self.input_shape[0]*self.input_shape[1]*self.input_shape[2], activation="relu"))
        model.add(layers.Reshape(self.input_shape))
        model.add(layers.Conv1D(64, kernel_size=1, strides=1, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(128, kernel_size=1, strides=1, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(1024, kernel_size=1, strides=1, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1DTranspose(512, kernel_size=2, strides=2, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1DTranspose(256, kernel_size=2, strides=2, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1DTranspose(self.input_shape[2], kernel_size=2, strides=2, padding="same", activation="tanh"))
        z = layers.Input(shape=(self.latent_dim,))
        point_cloud = model(z)
        return keras.models.Model(z, point_cloud)

    def build_discriminator(self):
        model = keras.models.Sequential()
        model.add(layers.Conv1D(64, kernel_size=1, strides=1, input_shape=self.input_shape, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(128, kernel_size=1, strides=1, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv1D(1024, kernel_size=1, strides=1, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=(2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        point_cloud = layers.Input(shape=self.input_shape)
        validity = model(point_cloud)
        return keras.models.Model(point_cloud, validity)

    def train(self, X_train, epochs, batch_size, sample_interval):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Select a random batch of point clouds
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            point_clouds = X_train[idx]

            # Sample noise and generate a batch of new point clouds
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_point_clouds = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(point_clouds, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_point_clouds, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample noise and generate a batch of new point clouds
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid)

            # If at sample interval, print metrics and generate point cloud samples
            if epoch % sample_interval == 0:
                print(f"Epoch {epoch}, Discriminator Loss: {d_loss[0]}, Discriminator Accuracy: {100*d_loss[1]}, Generator Loss: {g_loss}")
                self.sample_point_clouds(epoch)

    def sample_point_clouds(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_point_clouds = self.generator.predict(noise)
        gen_point_clouds = 0.5 * gen_point_clouds + 0.5

        fig = plt.figure(figsize=(10, 10))
        for i in range(r*c):
            ax = fig.add_subplot(r, c, i+1, projection='3d')
            ax.scatter(gen_point_clouds[i, :, 0], gen_point_clouds[i, :, 1], gen_point_clouds[i, :, 2])
            ax.axis('off')
        plt.savefig(f"epoch_{epoch}.png")
        plt.close()

