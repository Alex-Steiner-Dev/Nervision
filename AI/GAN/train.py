import matplotlib.pyplot as plt
from keras.optimizers import Adam
from dataset import parse_dataset
from GAN import *
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0][0]

def plotVoxel(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
    plt.savefig("generation.png")

x_train = np.array(train_data).reshape(1,32,32,32,1)

loss_fn = 'binary_crossentropy'
optimizer = Adam(lr=0.0002, beta_1=0.5)

G = build_generator()
D = build_discriminator()

# Compile the discriminator
D.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

# Compile the generator
G.compile(loss=loss_fn, optimizer=optimizer)

epochs = 1000
z_size = 200
batch_size = 1

for epoch in range(epochs):
    Z = np.random.normal(0, 0.33, (1, 1, 1, 1, 200))

    real_labels = np.ones((batch_size, 1, 1, 1, 1))
    fake_labels = np.zeros((batch_size, 1, 1, 1, 1))

    d_real_loss = D.train_on_batch(x_train, real_labels)

    fake = G.predict(Z)

    d_fake_loss = D.train_on_batch(fake, fake_labels)
        
    d_loss = d_real_loss[0] + d_fake_loss[0]

    d_real_acu = (d_real_loss[1] >= 0.5)
    d_fake_acu = (d_fake_loss[1] <= 0.5)
    d_total_acu = (d_real_acu + d_fake_acu) / 2

    if d_total_acu <= 0.8:
        D.trainable = True
        D.train_on_batch(x_train, real_labels)
        D.train_on_batch(fake, fake_labels)

    # Train the generator
    Z = np.random.normal(0, 0.33, (1, 1, 1, 1, 200))
    real_labels = np.ones((batch_size, 1, 1, 1, 1))
    g_loss = G.train_on_batch(Z, real_labels)

    # Update learning rates
    G_lr = max(0.0025 * (epochs - epoch) / epochs, 0)
    D_lr = max(0.01 * (epochs - epoch) / epochs, 0)
    K.set_value(G.optimizer.lr, G_lr)
    K.set_value(D.optimizer.lr, D_lr)