import trimesh
import trimesh
import numpy as np
import pyvista as pv


mesh = trimesh.load("dataset/f8d4e335655da8855e1d47baa7986b2b.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=2048)
point_cloud_array = np.array(vertices, dtype=np.float32)

mesh = pv.PolyData(point_cloud_array).delaunay_3d(.02).extract_geometry()

p = pv.Plotter()
p.add_mesh(mesh)
p.export_gltf("model.gltf")

pv.PolyData(point_cloud_array).plot()