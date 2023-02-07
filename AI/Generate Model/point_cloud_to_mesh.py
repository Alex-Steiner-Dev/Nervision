import numpy as np
import trimesh

def point_cloud_to_mesh_obj(points, file_name):
    cloud = trimesh.PointCloud(points)
    print(cloud)
    mesh = cloud.process()
    mesh.export("file_name" + '.obj', file_type='obj')

