import glob
import os
import sys
from mesh_to_point_cloud import *

from tqdm import tqdm
from time import sleep

modelName = "model.ape"
subDir = glob.glob("../Data/*")

def train():
    bar = tqdm(range(0, len(subDir)), desc = "Training Model")            

    open(modelName, "w").close()
    file = open(modelName, "a")

    for i in subDir:            
        three_d_model = glob.glob(i + "/*.obj")
        description = glob.glob(i + "/*.txt")
            
        mesh_to_point_cloud(three_d_model[0], i)
        point_cloud = glob.glob(i + "/*.pc")

        file.write("# " + os.path.basename(three_d_model[0]) + "\n")

        with open(description[0]) as description_file:
            file.write("// " + description_file.read() + "\n")

        with open(point_cloud[0]) as three_d_model_file:
            file.write(three_d_model_file.read())

        for j in bar:
            sleep(1)

train()