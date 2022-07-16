from re import L
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import igraph as ig
from plyfile import PlyData
from utils import *
from solver import *

PATH = "/Users/zqlwmatt/project/CV-project/Project3-Laplacian_Surface_Editing/"


handle = [14651, -0.0123491, 0.153911, -0.0264322]
ROI_vertices = [15692, 7357, 9877, 28992]

if __name__ == "__main__":
    print("Launch: Laplacian mesh editing")
    plydata = PlyData.read(PATH+'data/bun_zipper.ply')
    vertex, face = plyToNumpy(plydata)

    # 提取 ROI 重建图
    g = construct_graph(vertex, face)
    border_idx = get_border(g, ROI_vertices)
    vertices_idx = get_subgraph_vertices(g, handle[0], border_idx)
    vertices_idx += border_idx
    print(sorted(vertices_idx))

    sub_g = g.induced_subgraph(vertices_idx)
    n = len(vertices_idx)
    # print(V)

    # 全局/局部 标号映射
    absToSub = {}
    subToAbs = {}
    for i in range(n):
        k = sub_g.vs[i]['index']
        subToAbs[i], absToSub[k] = k, i

    # differential matrix

    V = np.vstack(tuple(vertex[idx] for i in range(n)))
    Delta = differential_matrix(sub_g, V)
    

    # ply_vertex, ply_face = numpyToPly(vertex, face)
    # PlyData([ply_vertex, ply_face], text=True).write(PATH+'test.ply')