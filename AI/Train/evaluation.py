import glob
from mesh_to_point_cloud import *
import open3d as o3d
from tqdm import tqdm
from time import sleep
import numpy as np
from scipy.spatial import distance

modelName = "../AI/Train/model.ape"
evaluationDir = glob.glob("../Evaluation/*.obj")

maxScore = 100

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
        whatisIt = ""
        with open(modelName, 'r') as file:
            while (line := file.readline().rstrip()):
                if line[0] == "/":
                    newModel = True
                    open("point_cloud.pc", "w").close()

                elif line[0] == "#":
                    whatisIt = line.replace("# ", "").replace(".obj", "")
                    newModel = False
                elif line == "|": 
                    pointCloud1 = np.loadtxt("../Evaluation/point_cloud.pc")
                    pointCloud2 = np.loadtxt("point_cloud.pc")

                    pointCloud1_split = np.array_split(pointCloud1, 100)
                    pointCloud2_split = np.array_split(pointCloud2, 100)

                    for l,m in zip(pointCloud1_split, pointCloud2_split):
                        distances = distance.cdist(l, m)

                        if (distances <= 300).any():
                            score+=1
                    
                                        
                    for j in bar:
                        sleep(1)
                        
                    print(f"The probability of being a {whatisIt} is { score / maxScore  * 100 }%")

                    score = 0

                elif newModel == True:
                  with open("point_cloud.pc", "a") as f:
                        f.write(line + "\n")

                count+=1