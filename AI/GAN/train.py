import matplotlib.pyplot as plt
from dataset import parse_dataset
from VAE import *
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()
train_data = np.array(train_data)

resolutions = []
box_size = 32

print("Training...")

autoencoder = VAE(box_size=box_size, resolutions=resolutions).build_vae()

history = autoencoder.fit(train_data, train_data, epochs=1000, batch_size=100)

autoencoder.save('autoencoder.h5')

print("Done!")