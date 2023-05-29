import trimesh
import numpy as np
import json
import pyvista as pv
from mesh_generation import *

f = open("captions.json")
data = json.load(f)

def simplify_mesh(mesh, n):
    # Compute the target reduction ratio
    target_reduction = 1.0 - (n / len(mesh.points))
    simplified_mesh = mesh.decimate(target_reduction=target_reduction, inplace=True)

    print(f"Number of vertices: {len(simplified_mesh.points)}")
    print(f"Number of faces: {simplified_mesh.n_cells}")

    simplified_mesh.plot()

    return simplified_mesh

a = 0
for i, itObject in enumerate(data):
    obj_path = "dataset/" + itObject['mid'] + ".obj"
    simplified_mesh = simplify_mesh(pv.read(obj_path), 2048)

    if simplified_mesh.n_cells > a:
        a = simplified_mesh.n_cells

print(a)

f.close()