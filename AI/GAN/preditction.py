import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from GAN import *

vae = VAE(box_size=32)

autoencoder, decoder = vae.build_vae()

autoencoder.load_weights("vae.h5")

z = np.random.normal(size=(1, 8, 8, 8, 120))
generated_sample = decoder.predict(z)

prediction = generated_sample.reshape(32, 32, 32)

def plotVoxel(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
    plt.show()

plotVoxel(prediction)