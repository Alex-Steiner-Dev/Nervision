import h5py
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from dataset import parse_dataset

autoencoder = load_model('autoencoder.h5')

z = parse_dataset()[0]
z = z.reshape([-1, 32, 32, 32, 1])

voxels = autoencoder.predict(z)
voxels = voxels.reshape(32, 32, 32)

def plotVoxel(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
    plt.show()

plotVoxel(voxels)