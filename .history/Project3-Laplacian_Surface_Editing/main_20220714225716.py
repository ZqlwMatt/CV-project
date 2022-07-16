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
    face = plydata['face'].data
    triangles = np.vstack(face['vertex_indices'])
    n, m = vertex.shape[0], face.shape[0]
    # vertex[idx] vectex['x'] vertex['y'] vertex['z'] face[0]

    ply_vertex = np.zeros((n, 3))

    for i in range(n):
        ply_vertex[i] = tuple(vertex[i])[:3]
    ply_vertex = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print(ply_vertex.shape)
    ply_vertex = PlyElement.describe(ply_vertex, 'vertex')
    PlyData([ply_vertex, face], text=True).write(PATH+'test.ply')