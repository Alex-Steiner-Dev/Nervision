import trimesh
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("model.h5")

def predict(input_points):
    input_points = input_points[np.newaxis, ...]
    predictions = model.predict(input_points)
    return predictions.argmax(axis=-1)[0]

def detect(mesh_path):
    mesh = trimesh.load(mesh_path)
    points = mesh.sample(NUM_POINTS)
    prediction = predict(points)
    return prediction

# Example usage
detection = detect("../Data/bathtub/test/bathtub_0107.off")
print("Detected class:", detection)
