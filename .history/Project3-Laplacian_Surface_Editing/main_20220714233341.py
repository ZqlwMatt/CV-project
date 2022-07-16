import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import igraph as ig
from plyfile import PlyData
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
    vertex, face = readPly()
    

    # ndarray 转 ply 类型
    ply_vertex, ply_face = numpyToPly(vertex, face)
    PlyData([ply_vertex, ply_face], text=True).write(PATH+'test.ply')