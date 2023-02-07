import numpy as np

modelName = "../Train/model.ape"

def generate_model(prompt):
    pass

def generate_single(word):
    foundModel = False
    model = []
    with open(modelName, "r") as file:
        while (line := file.readline()):
            if word in line:
                foundModel = True
            elif foundModel:
                model.insert(line)
            if line[0] == "#" and foundModel:
                break

    