import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# Load the image using OpenCV
gray = cv2.imread('depth.png')

# Generate a 3D array of X, Y, and Z values
X, Y = np.meshgrid(range(gray.shape[1]), range(gray.shape[0]))
Z = gray

# Combine X, Y, and Z into a single array
points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Write the point cloud to a PLY file
with open("point_cloud.ply", "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex {}\n".format(points.shape[0]))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for point in points:
        f.write("{} {} {}\n".format(*point))