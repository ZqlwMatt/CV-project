import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import igraph as ig
from plyfile import PlyData, PlyElement

PATH = "/Users/zqlwmatt/project/CV-project/Project3-Laplacian_Surface_Editing/"

handle_vertices = {
    14651: [-0.0123491, 0.153911, -0.0264322]
}

if __name__ == "__main__":
    print("Launch: Laplacian mesh editing")
    plydata = PlyData.read(PATH+'data/bun_zipper.ply')
    vertex = plydata['vertex'].data
    triangles = np.vstack(plydata['face'].data['vertex_indices'])
    n, m = vertex.shape[0], triangles.shape[0]
    # tri_data = plydata['face'].data[0]
    # vertex[idx] vectex['x'] vertex['y'] vertex['z'] face[0][0]
    # vertex =
    n_vertex = np.zeros((n, 3))
    # for i in range(n):
    #     n_vertex[i] = tuple(vertex[i])[:3]
    print(triangles)