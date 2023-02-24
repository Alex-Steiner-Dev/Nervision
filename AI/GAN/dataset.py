import trimesh
import glob
import numpy as np

DATA_DIR = "../Data/*"

def parse_dataset():
    print("Loading dataset...")
    
    objects = []

    folders = glob.glob(DATA_DIR)

    for i, folder in enumerate(folders):
        train_files = glob.glob(folder + "/train/*.off")

        point_cloud = trimesh.load(train_files[0]).sample(32)

        x, y, z = np.meshgrid(np.linspace(0, 1, 32),
                      np.linspace(0, 1, 32),
                      np.linspace(0, 1, 32))
        grid = np.stack([x, y, z], axis=-1)

        # Compute the Euclidean distances between each point and each voxel
        distances = np.linalg.norm(point_cloud[:, None, None, :] - grid[None, :, :, :], axis=-1)

        # Assign each point to the closest voxel
        voxels = np.argmin(distances, axis=1)

        objects.append(voxels)

    return objects