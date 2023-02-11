import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("point_cloud_classifier.h5")

DATA_DIR = "../Data/*"
CLASS_MAP = {0: "Bathtub", 1:"Bed", 2:"Chair", 3:"Desk", 4:"Dresser", 5:"Monitor", 6:"Night Stand", 7:"Sofa", 8:"Table", 9:"Toilet"}

def parse_dataset(num_points=2048):
    test_points = []
    test_labels = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))

        test_files = glob.glob(folder + "/test/*")

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(test_points),
        np.array(test_labels),
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

test_points, test_labels = parse_dataset(NUM_POINTS)

test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
plt.show()