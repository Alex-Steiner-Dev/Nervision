import open3d as o3d
import numpy as np

# Load the image
image = o3d.io.read_image("rgb.jpg")

# Convert the image to a numpy array
np_image = np.asarray(image)

# Convert the numpy array to a voxel representation
voxel_grid = o3d.geometry.VoxelGrid.create_from_depth_image(np_image, voxel_size=0.05)

# Visualize the voxel grid
o3d.visualization.draw_geometries([voxel_grid])