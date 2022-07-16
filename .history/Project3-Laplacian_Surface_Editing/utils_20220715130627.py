from re import S
import numpy as np
import igraph as ig
import networkx as nx
from plyfile import PlyData, PlyElement

def plyToNumpy(plydata):
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
    return n_vertex, n_face


def numpyToPly(vertex, face):
    """
    ndarray 转 ply 类型
    """
    ply_vertex = np.empty(vertex.shape[0], dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_face   = np.empty(face.shape[0], dtype = [('vertex_indices', 'i4', (3,))])
    ply_vertex['x'] = vertex[:, 0]
    ply_vertex['y'] = vertex[:, 1]
    ply_vertex['z'] = vertex[:, 2]
    ply_face['vertex_indices'] = face
    ply_vertex = PlyElement.describe(ply_vertex, 'vertex')
    ply_face = PlyElement.describe(ply_face, 'face')
    return ply_vertex, ply_face


def construct_graph(vertex, face):
    """
    从三角网格建图
    """
    n, m = vertex.shape[0], face.shape[0]
    g = ig.Graph(n)
    edges = set()
    for i in range(m):
        tri = sorted(face[i])
        edges.add((tri[0], tri[1]))
        edges.add((tri[1], tri[2]))
        edges.add((tri[0], tri[2]))
    g.add_edges(edges)
    return g

def get_border(g, ROI_vertices):
    border = []
    n = len(ROI_vertices)
    for i in range(n):
        segment = g.get_shortest_paths(ROI_vertices[i], ROI_vertices[(i+1)%n], output='vpath')[0]
        border.extend(segment[0:-1])
    return border

def get_components(g, vertex, border = None):
    """
    求连通块，igraph 没有方法，需要用 networkx
    """
    g1 = g.to_networkx().copy()
    g1.remove_nodes_from(border)
    return nx.node_connected_component(g1)