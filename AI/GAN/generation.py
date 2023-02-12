import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model
import pyvista as pv

input_shape = (2048,)

inputs = Input(shape=input_shape)

output = Dense(3, activation='linear')(inputs)

model = Model(inputs=inputs, outputs=output)

input_data = np.random.random((2048, 2048))

point_cloud = model.predict(input_data)

points = pv.PolyData(point_cloud)
points.plot()