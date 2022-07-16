# main.py: read pointcloud, call the core algorithm, show the result.
# The bunny pointcloud model is from http://graphics.stanford.edu/data/3Dscanrep/#uses
import numpy as np
import open3d as o3d
import lse_core

mesh = o3d.io.read_triangle_mesh('meshes/bun_zipper_res2.ply')

new_pts, new_graph_indices = lse_core.LSE(mesh, 0.025, -0.025, 0)

new_pcloud = o3d.geometry.PointCloud()
new_pcloud.points = o3d.utility.Vector3dVector(new_pts[new_graph_indices])
new_pcloud.paint_uniform_color([0.9, 0, 0])

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
point_cloud.paint_uniform_color([0.8, 0.8, 0.8])

o3d.visualization.draw_geometries([point_cloud, new_pcloud])
