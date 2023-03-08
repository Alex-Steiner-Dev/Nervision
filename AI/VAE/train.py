from VAE import VAE
from dataset import parse_dataset
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

noisy, x_train = parse_dataset()
noisy = np.array(noisy)
x_train = np.array(x_train)

box_size = 32

print("Training...")

autoencoder = VAE(box_size=box_size).build_vae()

history = autoencoder.fit(noisy, x_train, epochs=1000, batch_size=100)

autoencoder.save('../TrainedModels/autoencoder.h5')

print("Done!")