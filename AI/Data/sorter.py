import os 
import glob
import shutil

data_dir = "../ShapeNet/"
count = 12717

"""
for folder in glob.glob("*"):
    for sub_folder in glob.glob(folder + "/*"):
        model = glob.glob(sub_folder + "/*.obj")[0]

        os.rename(model, sub_folder + "/model_" + str(count) + ".obj")
        shutil.move(sub_folder + "/model_" + str(count) + ".obj", data_dir + "model_" + str(count) + ".obj")

        count+=1
"""
for i in range(count):
    with open(data_dir + "description_" + str(i) + ".txt", "w") as f:
        pass