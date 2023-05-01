import json
import objaverse
import shutil
import os

f = open("captions.json")
data = json.load(f)

for i in data:
    objects = objaverse.load_objects(
        uids=[i]
    )

f.close()

for path, subdirs, files in os.walk("glbs/"):
    for name in files:
        print()
        shutil.move(os.path.join(path, name), "dataset/" + name)