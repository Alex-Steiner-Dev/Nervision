import numpy as np
from point_cloud_to_mesh import *

modelName = "../AI/Train/model.ape"

def generate_model(prompt):
    generate_single(prompt[0][0])

def generate_single(word):
    foundModel = False
    points = np.array([])

    with open(modelName, "r") as file:
        while (line := file.readline()):
            if line[0] == "|" and foundModel:
                point_cloud_to_mesh_obj(points)
                break

            elif foundModel:
                new_point = np.array([float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2])])
                points = np.concatenate((points, new_point), axis=0)

            elif word in line and not line[1] == "#":
                foundModel = True