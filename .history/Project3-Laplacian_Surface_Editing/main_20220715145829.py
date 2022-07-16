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

    # 建图 提取ROI
    g = construct_graph(vertex, face)
    border = get_border(g, ROI_vertices)
    vertices = get_subgraph_vertices(g, handle[0], border)
    vertices += border
    sub_g = g.induced_subgraph(vertices)

    # 标号对应关系
    absToSub = {}
    subToAbs = {}
    n = len(vertices)
    for i in range(n):
        k = sub_g.vs[i]['index']
        subToAbs[i], absToSub = k
        absToSub[k] = i

    ply_vertex, ply_face = numpyToPly(vertex, face)
    PlyData([ply_vertex, ply_face], text=True).write(PATH+'test.ply')