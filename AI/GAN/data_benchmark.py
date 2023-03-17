import trimesh
import glob
import numpy as np

def parse_dataset(root, npoints):
    objects = []

    folders = glob.glob(root)

    for i, folder in enumerate(folders):
        train_points = []
        
        train_files = glob.glob(folder + "/train/*")

        for f in train_files:
            train_points.append(trimesh.load(f).sample(npoints))

            objects.append(train_points)

    return np.array(objects)