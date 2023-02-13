import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.models import load_model
import pyvista as pv

model = load_model("../Train/point_cloud_classifier.h5")

input_data = np.random.random((2048, 2048))

point_cloud = model.predict(input_data)

points = pv.PolyData(point_cloud)
points.plot()