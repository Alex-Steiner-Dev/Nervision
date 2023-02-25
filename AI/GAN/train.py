from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
import matplotlib.pyplot as plt
from dataset import parse_dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras import backend as K
K.clear_session()

train_data = parse_dataset()[0]

box_size = 32

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])

input_img = Input(shape=(box_size, box_size, box_size, 1))

x = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(x)
encoder = MaxPooling3D((2, 2, 2), padding='same')(x)

x = Convolution3D(128, (3, 3, 3), activation='relu', padding='same')(encoder)
x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(64, (3, 3, 3), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(32, (3, 3, 3), activation='relu', padding='same')(x)
decoder = Convolution3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoder)

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(train_data, train_data, epochs=100, batch_size=32, validation_split=0.1)

decoder.save('autoencoder.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()