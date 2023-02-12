import trimesh
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("point_cloud_classifier.h5")
CLASS_MAP = {0: "bathtub", 1:"bed", 2:"chair", 3:"desk", 4:"dresser", 5:"monitor", 6:"night stand", 7:"sofa", 8:"table", 9:"toilet"}

def evaluate(path):
    points = trimesh.load(path).sample(2048)
    points = np.expand_dims(points, axis=0)

    preds = model.predict(points)

    temp = 0
    for i in preds:
        for j in i:
            if j > temp:
                temp = j

    preds = tf.math.argmax(preds, -1)

    print(f"The model given is to {round(temp * 100, 2)}% a {CLASS_MAP[int(preds.numpy())]}")

evaluate("../Data/night_stand/test/night_stand_0286.off")