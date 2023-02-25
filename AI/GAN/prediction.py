import h5py
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


autoencoder = load_model('autoencoder.h5')

z = np.random.normal(size=(64, 64,64))
z = z.reshape([-1, 64, 64, 64, 1])

voxels = autoencoder.predict(z)
voxels = voxels.reshape(64, 64, 64)

print(voxels)

x, y, z = np.indices((64, 64, 64))
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.voxels(voxels, facecolors='#5DADE2', edgecolors='#34495E')
plt.show()

