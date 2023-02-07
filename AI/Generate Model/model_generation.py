import numpy as np

modelName = "../Train/model.ape"

def generate_model(prompt):
    pass

def generate_single(word):
    foundModel = False
    points = np.array([[0,0,0]])

    with open(modelName, "r") as file:
        while (line := file.readline()):
            if line[0] == "|" and foundModel:
                break

            elif foundModel:
                print(line)
                b = np.array([[float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2]) ]])
                points = np.concatenate((points, b), axis=0)

            elif word in line and not line[1] == "#":
                foundModel = True

generate_single("chair")