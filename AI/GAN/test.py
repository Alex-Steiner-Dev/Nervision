import trimesh
import trimesh
import numpy as np
import pyvista as pv


mesh = trimesh.load("dataset/29876b641047897f9bbef973cc1034f9.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=2048)
point_cloud_array = np.array(vertices, dtype=np.float32)

mesh = pv.PolyData(point_cloud_array).delaunay_3d(.045).extract_geometry()

p = pv.Plotter()
p.add_mesh(mesh)
p.export_gltf("model.gltf")
p.show()
