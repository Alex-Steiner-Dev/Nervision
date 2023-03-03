from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, Conv2DTranspose
import numpy as np

class GAN:
    def __init__(self, latent_dim, x_train, y_train):
        self.latent_dim = latent_dim

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

        self.x_train = x_train
        self.y_train = y_train

    def build_generator(self):
        model = Sequential(name="Generator") 

        n_nodes = 256 * 3
        model.add(Dense(n_nodes, input_dim=self.latent_dim,))
        model.add(Reshape((256, 3)))
        
        model.add(Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), padding='same'))
        model.add(ReLU())
                                
        model.add(Conv2DTranspose(filters=1025, kernel_size=(4,4), strides=(2,2), padding='same'))
        model.add(ReLU())
        
        model.add(Conv2DTranspose(filters=2048, kernel_size=(4,4), strides=(2,2), padding='same'))
        model.add(ReLU())
        
        model.add(Conv2D(filters=3, kernel_size=(5,5), activation='tanh', padding='same'))

        return model
    
    def build_discriminator(in_shape=(2048,3)):
        model = Sequential(name="Discriminator") 

        model.add(Conv2D(filters=2048*2, kernel_size=(4,4), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(filters=2048*4, kernel_size=(4,4), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(filters=2048*6, kernel_size=(4,4), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Flatten())
        model.add(Dropout(0.3)) 
        model.add(Dense(1, activation='sigmoid')) 
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
        
        return model
    
    def build_gan(self):
        self.discriminator.trainable = False
        
        model = Sequential(name="DCGAN")
        model.add(self.generator) 
        model.add(self.discriminator)
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        return model
    
    def latent_vector(latent_dim):
        latent_input = np.random.randn(latent_dim)
        latent_input = latent_input.reshape(1, latent_dim)

        return latent_input

    def fake_samples(self):
        latent_output = self.latent_vector(self.latent_dim, 1)
        
        X = self.generator.predict(latent_output)
        
        y = self.y_train[np.random.randint(0, len(self.y_train) - 1)]

        return X, y
    
    def real_samples(self):
        random = np.random.randint(0, len(self.y_train) - 1)

        X = self.x_train[random]

        y = np.ones((1, 1))

        return X, y
    
    def train(self, epochs):
        for i in range(epochs):
            x_real, y_real = self.real_samples(self.x_train, self.y_train)
            x_fake, y_fake = self.fake_samples(self.generator, self.latent_dim)
            
    
            X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = self.discriminator.train_on_batch(X, y)
        
            x_gan = self.latent_vector(self.latent_dim)
            y_gan = np.ones((1, 1))
            
            generator_loss = self.gan_model.train_on_batch(x_gan, y_gan)
            
            if i % 100 == 0:
                print("Epoch number: ", i)
                print("*** Training ***")
                print("Discriminator Loss ", discriminator_loss)
                print("Generator Loss: ", generator_loss)