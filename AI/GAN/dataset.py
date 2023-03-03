import glob
import numpy as np
import logging
import trimesh

from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("trimesh").setLevel(logging.ERROR)

DATA_DIR = "../Data/ShapeNet"

def parse_dataset():
    print("Loading dataset...")
    
    point_clouds = []
    labels = []

    meshes = glob.glob(DATA_DIR + "/*.obj")
    label_files = glob.glob(DATA_DIR + "/*.txt")
        
    for i in tqdm(range(len(meshes))):
        point_cloud = trimesh.load(meshes[i], force='mesh').sample(2048)
        point_clouds.append(point_cloud)

    for i in tqdm(range(len(label_files))):
        label_files.append(open(label_files[i], 'r').readlines())

    print("Done!")

    return point_clouds, labels
