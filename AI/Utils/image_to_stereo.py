# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read two input images as grayscale images
imgL = cv2.imread('input.png',0)
imgL = cv2.imread('input.png',0)

# Initiate and StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# compute the disparity map
disparity = stereo.compute(imgL, imgL)
plt.imshow(disparity,'gray')
plt.axis('off')
plt.savefig('out.png', bbox_inches='tight', pad_inches=0)
plt.show()