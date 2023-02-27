from dataset import parse_dataset
from GAN import *
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

x_train = parse_dataset()
x_train = np.array(x_train)

box_size = 64 

print("Training...")

gan = GAN(box_size=box_size)
gan.train(x_train=x_train, epochs=500)

print("Done!")
