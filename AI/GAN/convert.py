import json
import objaverse
import shutil
import os

import open3d as o3d

f = open("captions.json")
data = json.load(f)

uids = []

for i in data:
    uids.append(i)

f.close()

for i,item in enumerate(os.listdir("dataset/")):
    if item[len(item) - 1] == "b":
        mesh = o3d.io.read_triangle_mesh("dataset/" + uids[i] + ".glb")
        o3d.io.write_triangle_mesh("dataset/" + uids[i] + ".obj", mesh)


for item in os.listdir("dataset/"):
    if not item[len(item) - 1] == "j":
        os.remove("dataset/" + item)