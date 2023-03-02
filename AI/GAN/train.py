from dataset import parse_dataset
from GAN import *
import numpy as np

x_train, y_train = parse_dataset()
x_train = np.array(x_train)
y_train = np.array(y_train)

print("Training...")

print("Done!")
