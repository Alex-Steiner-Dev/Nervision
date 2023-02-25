import matplotlib.pyplot as plt
from dataset import parse_dataset
from VAE import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0]

resolutions = [4, 8, 16]
box_size = 64

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])

autoencoder = VAE(box_size=box_size, resolutions=resolutions).build_vae()

history = autoencoder.fit(train_data, train_data, epochs=200, batch_size=1)

autoencoder.save('autoencoder.h5')