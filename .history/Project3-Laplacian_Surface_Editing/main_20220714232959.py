import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import igraph as ig
from plyfile import PlyData, PlyElement
from utils import *

PATH = "/Users/zqlwmatt/project/CV-project/Project3-Laplacian_Surface_Editing/"

handle_vertices = {
    14651: [-0.0123491, 0.153911, -0.0264322]
    # ... 
}
ROI_vertices = [15692, 7357, 9877, 28992]

if __name__ == "__main__":
    print("Launch: Laplacian mesh editing")
    plydata = PlyData.read(PATH+'data/bun_zipper.ply')
    """
    n_vertex, n_face : 转换为 ndarray 的点集 / 面集
    n, m : 点数 / 面数
    """
    vertex = plydata['vertex'].data
    face = plydata['face'].data
    n, m = vertex.shape[0], face.shape[0]
    # vertex[idx] vectex['x'] vertex['y'] vertex['z'] face[0]
    n_vertex = np.zeros((n, 3))
    for i in range(n):
        n_vertex[i] = tuple(vertex[i])[:3]
    n_face = np.vstack(face['vertex_indices'])

    

    # ndarray 转 ply 类型
    ply_vertex = np.empty(vertex.shape[0], dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_face = np.empty(n_face), dtype = [('vertex_indices', 'i4', (3,))])
    ply_vertex['x'] = n_vertex[:, 0]
    ply_vertex['y'] = n_vertex[:, 1]
    ply_vertex['z'] = n_vertex[:, 2]
    ply_face['vertex_indices'] = n_face
    ply_vertex = PlyElement.describe(ply_vertex, 'vertex')
    ply_face = PlyElement.describe(ply_face, 'face')
    PlyData([ply_vertex, ply_face], text=True).write(PATH+'test.ply')