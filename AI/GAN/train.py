import matplotlib.pyplot as plt
from dataset import parse_dataset
from GAN import *
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0]
train_data = np.array(train_data)

box_size = 32

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])

vae = VAE(box_size=32, resolutions=[1]).build_vae()[0]
vae.fit(train_data, train_data, epochs=5000, batch_size=1)

vae.save("vae.h5")