
import glob
import os
import linecache
from mesh_to_point_cloud import *

from tqdm import tqdm
from time import sleep

modelName = "model.ape"
evaluationDir = glob.glob("../Evaluation/*.obj")

tollerances = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
maxScore = 17377500

def evalutate():
    bar = tqdm(range(0, len(evaluationDir)), desc = "Evaluating Model") 

    file = open(modelName, "r") 

    for i in evaluationDir:      
        mesh_to_point_cloud(i, "../Evaluation") 
        lines = ""

        with open(modelName, "r") as f:
            lines = f.readlines()

        newModel = False
        count = 0
        score = 0

        with open(modelName, 'r') as file:
            while (line := file.readline().rstrip()):
                if line[0] == "/":
                    newModel = True
                elif line[0] == "#":
                    newModel = False
                elif newModel == True:
                    pos_x = float(line.split()[0])
                    pos_y = float(line.split()[0])
                    pos_z = float(line.split()[0])

                    model_pos_x = float(lines[count].split()[0])
                    model_pos_y = float(lines[count].split()[1])
                    model_pos_z = float(lines[count].split()[2])

                    for k in tollerances:
                        if pos_x - model_pos_x < k and pos_x - model_pos_x < k:
                            score+= tollerances[len(tollerances) - 1] - k
                        elif pos_y - model_pos_y < k and pos_y - model_pos_y < k:
                            score+= tollerances[len(tollerances) - 1] - k
                        elif pos_z - model_pos_z < k and pos_z - model_pos_z < k:
                            score+= tollerances[len(tollerances) - 1] - k

                count+=1

        for j in bar:
            sleep(1)

        print(f"The probability of being a chair is {maxScore / score * 100}")
        
evalutate()