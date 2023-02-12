import trimesh
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("point_cloud_classifier.h5")

CLASS_MAP = {0: "Bathtub", 1:"Bed", 2:"Chair", 3:"Desk", 4:"Dresser", 5:"Monitor", 6:"Night Stand", 7:"Sofa", 8:"Table", 9:"Toilet"}

points = trimesh.load("../Data/desk/test/desk_0201.off").sample(2048)
print(points.shape)

preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

print("pred: {:}".format(CLASS_MAP[preds.numpy()]))