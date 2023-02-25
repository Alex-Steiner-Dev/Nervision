import h5py
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


autoencoder = load_model('autoencoder.h5')

z = np.random.normal(size=(32, 32,32))
z = z.reshape([-1, 32, 32, 32, 1])

voxels = autoencoder.predict(z, batch_size=1)
voxels = voxels.reshape(32, 32, 32)

x, y, z = np.indices((32, 32, 32))
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
plt.show()

