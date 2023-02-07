import pyrecon
import pypcd
import numpy as np

def point_cloude_to_mesh():
    pc = pypcd.PointCloud.from_path("path_to_point_cloud.pcd")

    mesh = pypcd.PoissonReconstruction(pc.pc_data[['x', 'y', 'z']].values)

    mesh.save_mesh("path_to_mesh.ply")


point_cloude_to_mesh()