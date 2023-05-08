import numpy as np
import pyvista as pv
import trimesh

mesh = trimesh.load("dataset\\40f1be4ede6113a2e03aea0698586c31.obj", force="mesh")
points = mesh.sample(100000)
point_cloud = np.array(points)

point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 4096, replace=False), :]


pv.PolyData(point_cloud).plot()