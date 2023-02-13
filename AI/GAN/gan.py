import trimesh
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyvista as pv

model = load_model("../Train/point_cloud_classifier.h5")

def evaluate(points):
    points = np.expand_dims(points, axis=0)
    preds = model.predict(points)

    temp = 0
    for i in preds:
        for j in i:
            print(j)
            if j > temp:
                temp = j

    preds = tf.math.argmax(preds, -1)

    if round(temp * 100, 2) <= 50:
        print("Fake")
    else:
        print("Real")

points = np.random.rand(2048,3)
point_cloud = pv.PolyData(points)
point_cloud.plot()
evaluate(points)