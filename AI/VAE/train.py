from dataset import *
import numpy as np
from GAN import *

from keras import backend as K
K.clear_session()

x_train = parse_dataset()
x_train = np.array(x_train)

x_train = x_train.reshape([-1,256,256,256,1])
box_size = 256

print("Training...")

epochs = 10000
batch_size = 1
save_interval = 100

train_gan(x_train,generator, discriminator, gan, epochs, batch_size, save_interval)

print("Done!")