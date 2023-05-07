import numpy as np
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import pyvista as pv
import trimesh

mesh = trimesh.load("dataset\\40f1be4ede6113a2e03aea0698586c31.obj", force="mesh")
points = mesh.sample(100000)
point_cloud = np.array(points)

point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], 6144, replace=False), :]

x_interp = interp1d(np.linspace(0, 1, 6144), point_cloud[:, 0], kind='cubic')
x_interp = interp1d(np.linspace(0, 1, 6144), point_cloud[:, 0], kind='nearest', fill_value='extrapolate')
y_interp = interp1d(np.linspace(0, 1, 6144), point_cloud[:, 1], kind='nearest', fill_value='extrapolate')
z_interp = interp1d(np.linspace(0, 1, 6144), point_cloud[:, 2], kind='nearest', fill_value='extrapolate')

point_cloud = np.zeros((100000, 3))
point_cloud[:, 0] = x_interp(np.linspace(0, 1, 100000))
point_cloud[:, 1] = y_interp(np.linspace(0, 1, 100000))
point_cloud[:, 2] = z_interp(np.linspace(0, 1, 100000))

