import matplotlib.pyplot as plt
from dataset import parse_dataset
from GAN import *
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0][0]
train_data = np.array(train_data).reshape([-1, 32, 32, 32, 1])

vae = VAE(box_size=32)

autoencoder, decoder = vae.build_vae()

autoencoder.fit(train_data, train_data, epochs=100000, batch_size=1)

z = np.random.normal(size=(1, 8, 8, 8, 120))
generated_sample = decoder.predict(z)

prediction = generated_sample.reshape(32, 32, 32)

def plotVoxel(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
    plt.show()

plotVoxel(prediction)