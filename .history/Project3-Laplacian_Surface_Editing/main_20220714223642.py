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
    triangles = np.vstack(face)
    n, m = vertex.shape[0], face.shape[0]
    # vertex[idx] vectex['x'] vertex['y'] vertex['z'] face[0]
    n_vertex = np.zeros((n, 3))
    # for i in range(n):
    #     n_vertex[i] = tuple(vertex[i])[:3]
    n_vertex, n_face = vertex, face
    print(triangles)
    n_vertex = PlyElement.describe(n_vertex, 'vertex')
    n_face = PlyElement.describe(n_face, 'face')
    PlyData([n_vertex, n_face], text=True).write(PATH+'test.ply')