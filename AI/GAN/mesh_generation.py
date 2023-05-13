import pyvista as pv

def generate_mesh(points):
    mesh = pv.PolyData(points).delaunay_3d(.045).extract_geometry()

    return mesh