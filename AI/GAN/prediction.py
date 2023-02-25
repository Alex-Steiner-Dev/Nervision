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


def saveFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='blue')
    plt.show()

saveFromVoxels(voxels)