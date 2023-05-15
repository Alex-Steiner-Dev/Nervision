import trimesh
import trimesh
import numpy as np
import pyvista as pv


mesh = trimesh.load("dataset/e501ac06884534b5d5f962f170e40e8e.obj", force="mesh")

vertices, _ = trimesh.sample.sample_surface(mesh, count=2048)
point_cloud_array = np.array(vertices, dtype=np.float32)

mesh = pv.PolyData(point_cloud_array).delaunay_3d(.02).extract_geometry()

p = pv.Plotter()
p.add_mesh(mesh)
p.export_gltf("model.gltf")

pv.PolyData(point_cloud_array).plot()