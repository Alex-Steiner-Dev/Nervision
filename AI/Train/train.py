import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

tf.random.set_seed(1234)

DATA_DIR = "../Data/*"

def parse_dataset(num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
    
        class_map[i] = folder.split("/")[-1]
  
        train_files = glob.glob(folder + "/train/*")
        test_files = glob.glob(folder + "/test/*")

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
    NUM_POINTS
)


print(CLASS_MAP)

num_classes = len(np.unique(train_labels))
train_points = train_points.reshape(train_points.shape[0], -1)
test_points = test_points.reshape(test_points.shape[0], -1)
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(train_points.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_points, train_labels, epochs=500, batch_size=64, validation_data=(test_points, test_labels))

test_loss, test_acc = model.evaluate(test_points, test_labels)
print('Test accuracy:', test_acc)

model.save("point_cloud_classifier.h5")