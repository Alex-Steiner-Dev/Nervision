import numpy as np
from point_cloud_to_mesh import *

modelName = "../AI/Train/model.ape"
iteration = 0

def generate_model(prompt):
    print(prompt[0][0])
    generate_single(prompt[0][0])

def generate_single(word):
    foundModel = False
    with open(modelName, "r") as file:
        while (line := file.readline()):
            if line[0] == "|" and foundModel:
                print(np.loadtxt(f"generation {iteration}.txt"))
                point_cloud_to_mesh_obj(np.loadtxt(f"generation {iteration}.txt"), "model.obj")
                break

            elif foundModel:
                with open(f"generation {iteration}.txt", "w") as f:
                    f.write(line + "\n")

            elif word in line and not line[1] == "#":
                foundModel = True