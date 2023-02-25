import matplotlib.pyplot as plt
from dataset import parse_dataset
from VAE import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0]
resolutions = [64, 128, 256, 512]
box_size = 1024

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])

autoencoder, encoder, decoder = VAE(box_size=box_size, resolutions=resolutions).build_vae()

history = autoencoder.fit(train_data, train_data, epochs=200, batch_size=1)

decoder.save('autoencoder.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()