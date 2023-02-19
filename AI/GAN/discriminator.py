import trimesh
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("../Train/point_cloud_classifier.h5")

model.summary()
CLASS_MAP = {0: "bathtub", 1:"bed", 2:"chair", 3:"desk", 4:"dresser", 5:"monitor", 6:"night stand", 7:"sofa", 8:"table", 9:"toilet"}

def evaluate(points):
    preds = model.predict(points)

    temp = 0
    for i in preds:
        for j in i:
            if j > temp:
                temp = j

    preds = tf.math.argmax(preds, -1)

    return (round(temp, 2), CLASS_MAP[int(preds.numpy())])