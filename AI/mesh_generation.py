import pyvista as pv

points = pv.read("1a6ad7a24bb89733f412783097373bdc.ply")
cloud = pv.PolyData(points)

volume = cloud.delaunay_3d(alpha=.02)

shell = volume.extract_geometry()
shell.plot()