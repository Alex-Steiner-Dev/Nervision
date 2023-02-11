import trimesh
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("point_cloud_classifier.h5")

# Preprocess a new point cloud
new_point_cloud = trimesh.load("../Data/bathtub/test/bathtub_0141.off").sample(2048)
new_point_cloud = new_point_cloud.reshape(1, -1) # Reshape the point cloud into a 2D array


predictions = loaded_model.predict(new_point_cloud)
predicted_class = np.argmax(predictions)

labels = {0: 'Data\\bathtub', 1: 'Data\\bed', 2: 'Data\\chair', 3: 'Data\\desk', 4: 'Data\\dresser', 5: 'Data\\monitor', 6: 'Data\\night_stand', 7: 'Data\\sofa', 8: 'Data\\table', 9: 'Data\\toilet'}

print("Predicted class label:", labels[predicted_class])
