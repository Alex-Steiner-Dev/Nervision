import json
import objaverse
import shutil
import os

import open3d as o3d

f = open("captions.json")
data = json.load(f)

path = "C:\\Users\\Marco\\.objaverse\\hf-objaverse-v1\\glbs"
uids = []

for i in data:
    uids.append(i)
    objects = objaverse.load_objects(
        uids=[i]
    )

f.close()

shutil.move(path, "./")

for path, subdirs, files in os.walk("glbs/"):
    for i, name in enumerate(files):
        shutil.move(os.path.join(path, name), "dataset/" + name)

