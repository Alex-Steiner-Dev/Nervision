import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

DATA_DIR = "../Data/*"

def parse_dataset(num_points=2048):
    test_points = []
    test_labels = []
    class_map = {}

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
    
        class_map[i] = folder.split("/")[-1]

        test_files = glob.glob(folder + "/test/*")

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(test_points),
        np.array(test_labels),
        class_map,
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

test_points, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)


test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

model = tf.keras.models.load_model('model')

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