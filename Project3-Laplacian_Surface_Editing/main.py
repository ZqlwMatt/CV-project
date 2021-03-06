import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from plyfile import PlyData
from utils import *

PATH = "/Users/zqlwmatt/project/CV-project/Project3-Laplacian_Surface_Editing/"

# 数据格式: [handle vertex] [idx, x, y, z] | [ROI vertices] [idx]
handle = [24092, -0.01134,  0.151374, -0.0242688]
ROI_vertices = [15617, 24120, 30216, 11236, 6973]
# handle = [14651, -0.0123491, 0.153911, -0.0264322]
# ROI_vertices = [15692, 7357, 9877, 28992]

# 约束系数: [0.3, 1.5]
alpha = 1.

if __name__ == "__main__":
    print("Launch: Laplacian mesh editing...")
    plydata = PlyData.read(PATH+'data/bun_zipper.ply')
    vertex, face = plyToNumpy(plydata)

    # 提取 ROI 子图
    g = construct_graph(vertex, face)
    border_idx = get_border(g, ROI_vertices)
    vertices_idx = get_subgraph_vertices(g, handle[0], border_idx) ; vertices_idx += border_idx
    sub_g = g.induced_subgraph(vertices_idx)
    n = len(vertices_idx)
    m = len(ROI_vertices) + 1
    
    # 全局/局部 标号映射
    absToSub = {}
    subToAbs = {}
    for i in range(n):
        k = sub_g.vs[i]['index']
        subToAbs[i], absToSub[k] = k, i
    print("Graph is constructed.")

    # Laplacian operator & differential matrix
    sub_V = np.vstack([vertex[subToAbs[i]] for i in range(n)]) # 按照 igraph subgraph 的顺序建立点矩阵
    # assert(0)
    D = np.diag(sub_g.degree(list(range(n))))
    Lap = np.array(sub_g.laplacian())
    L = np.linalg.inv(D).dot(Lap)
    Delta = np.dot(L, sub_V)
    # print(Delta)

    # 建立方程
    Lp = np.zeros((3*(n+m), 3*n))
    Lp[0*n : 1*n, 0*n : 1*n] = (-1) * L
    Lp[1*n : 2*n, 1*n : 2*n] = (-1) * L
    Lp[2*n : 3*n, 2*n : 3*n] = (-1) * L
    for i in range(n):
        # 邻接点
        Ni = sub_g.neighbors(i)
        Ni.append(i)
        Ni = np.array(Ni)
        siz = len(Ni)
        Ai = np.zeros((3*siz, 7))
        # 邻接仿射矩阵
        for j in range(siz):
            pos = sub_V[Ni[j]]
            Ai[j+0*siz] = np.array([pos[0],       0,  pos[2], -pos[1], 1, 0, 0])
            Ai[j+1*siz] = np.array([pos[1], -pos[2],       0,  pos[0], 0, 1, 0])
            Ai[j+2*siz] = np.array([pos[2],  pos[1], -pos[0],       0, 0, 0, 1])

        Ti = scipy.linalg.pinv(Ai)
        si, hi, ti = Ti[0], Ti[1:4], Ti[4:7] # ti 在这一部分没有使用
        Delta_ix = Delta[i, 0]
        Delta_iy = Delta[i, 1]
        Delta_iz = Delta[i, 2]

        T_delta = np.array([
              Delta_ix*   si - Delta_iy*hi[2] + Delta_iz*hi[1], # x 的T变换方程
              Delta_ix*hi[2] + Delta_iy*   si - Delta_iz*hi[0], # y 的T变换方程
            - Delta_ix*hi[1] + Delta_iy*hi[0] + Delta_iz*   si  # z 的T变换方程
        ])

        # 修改 L prime 矩阵
        Ni_idx = np.hstack((Ni, Ni+n, Ni+2*n))
        Lp[0*n+i, Ni_idx] += T_delta[0]
        Lp[1*n+i, Ni_idx] += T_delta[1]
        Lp[2*n+i, Ni_idx] += T_delta[2]
    
    b = np.array([])
    # 设置 handle 点的约束方程
    handle_vertices = np.array(handle[1:])
    b = np.append(b, alpha * handle_vertices) # x y z
    idx = absToSub[handle[0]]
    Lp[3*n+0, 0*n+idx] = alpha
    Lp[3*n+1, 1*n+idx] = alpha
    Lp[3*n+2, 2*n+idx] = alpha
    # 设置 roi 点的约束方程
    for i in range(m-1):
        b = np.append(b, alpha * vertex[ROI_vertices[i]])
        idx = absToSub[ROI_vertices[i]]
        Lp[3*(n+1)+i*3,   0*n+idx] = alpha
        Lp[3*(n+1)+i*3+1, 1*n+idx] = alpha
        Lp[3*(n+1)+i*3+2, 2*n+idx] = alpha
    
    b = np.hstack((np.zeros(3*n), b))
    Lp = scipy.sparse.coo_matrix(Lp)
    Vp = scipy.sparse.linalg.lsqr(Lp, b)[0]
    print('Equation is solved.')

    # modify deformation points
    for i in range(n):
        vertex[subToAbs[i]] = np.array([Vp[i], Vp[i+n], Vp[i+2*n]])
    ply_vertex, ply_face = numpyToPly(vertex, face)
    PlyData([ply_vertex, ply_face], text=True).write(PATH+'result.ply')
    print("finish!")
