from keras.models import load_model
import numpy as np

generator = load_model("gan.h5")

noise = np.random.normal(0, 1, (1, 100))
generated_cloud = generator.predict(noise)