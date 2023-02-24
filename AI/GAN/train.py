import numpy as np
import tensorflow as tf
from keras.activations import sigmoid
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K

from VAE import *
from dataset import parse_dataset

learning_rate_1 = 0.0001
learning_rate_2 = 0.005
momentum = 0.9
batch_size = 10
epoch_num = 150

def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def learning_rate_scheduler(epoch, lr):
    if epoch >= 1:
        lr = learning_rate_2
    return lr

if __name__ == '__main__':
    model = get_model()

    inputs = model['inputs']
    outputs = model['outputs']
    mu = model['mu']
    sigma = model['sigma']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']

    vae = model['vae']

    # kl_div = -0.5 * K.mean(1 + 2 * sigma - K.square(mu) - K.exp(2 * sigma))
    voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32') # + kl_div
    vae.add_loss(voxel_loss)

    sgd = SGD(lr = learning_rate_1, momentum = momentum, nesterov = True)
    vae.compile(optimizer = sgd, metrics = ['accuracy'])

    data_train = parse_dataset()[0]

    vae.fit(
        data_train,
        epochs = epoch_num,
        batch_size = batch_size,
    )

    vae.save_weights('vae.h5')
