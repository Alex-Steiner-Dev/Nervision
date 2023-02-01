import numpy as np
import cv2

# Load the image
img = cv2.imread("test.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a depth image
depth = gray.astype(np.float32) / 255.0

# Generate the point cloud
rows, cols = depth.shape
[X, Y] = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1))
Z = depth

# Stack the X, Y, and Z arrays to form the point cloud
point_cloud = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

print(point_cloud.shape)

from stl import mesh

# Create a mesh from the point cloud
number_of_triangles = (rows - 1) * (cols - 1)
triangles = np.zeros((number_of_triangles, 3, 3), dtype=np.float32)
triangle_index = 0
for i in range(rows - 1):
    for j in range(cols - 1):
        # First triangle
        triangles[triangle_index, 0, :] = point_cloud[i * cols + j, :]
        triangles[triangle_index, 1, :] = point_cloud[i * cols + j + 1, :]
        triangles[triangle_index, 2, :] = point_cloud[(i + 1) * cols + j, :]
        triangle_index += 1

        # Second triangle
        triangles[triangle_index, 0, :] = point_cloud[(i + 1) * cols + j + 1, :]
        triangles[triangle_index, 1, :] = point_cloud[(i + 1) * cols + j, :]
        triangles[triangle_index, 2, :] = point_cloud[i * cols + j + 1, :]
        triangle_index += 1


mesh = mesh.Mesh(triangles, remove_empty_areas=False)

# Save the mesh to a PLY file
mesh.save('point_cloud.ply', mode=mesh.Mesh.MODE_ASCII)