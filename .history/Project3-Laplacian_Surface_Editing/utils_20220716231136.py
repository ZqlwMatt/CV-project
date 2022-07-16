"""
This lib is mainly used for (1) transformation between plyfile and numpy, 
(2) graph construction and operator

"""
import numpy as np
import igraph as ig
import networkx as nx
from plyfile import PlyData, PlyElement

# python-plyfile guidline: https://github.com/dranjan/python-plyfile
def plyToNumpy(plydata):
    vertex = plydata['vertex'].data
    face = plydata['face'].data
    n = vertex.shape[0]
    # vertex[idx] vectex['x'] vertex['y'] vertex['z'] face[0]
    n_vertex = np.zeros((n, 3))
    for i in range(n):
        n_vertex[i] = tuple(vertex[i])[:3]
    n_face = np.vstack(face['vertex_indices'])
    return n_vertex, n_face


def numpyToPly(vertex, face):
    ply_vertex = np.empty(vertex.shape[0], 
                          dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_face   = np.empty(face.shape[0],
                          dtype = [('vertex_indices', 'i4', (3,))])
    ply_vertex['x'] = vertex[:, 0]
    ply_vertex['y'] = vertex[:, 1]
    ply_vertex['z'] = vertex[:, 2]
    ply_face['vertex_indices'] = face
    ply_vertex = PlyElement.describe(ply_vertex, 'vertex')
    ply_face   = PlyElement.describe(ply_face, 'face')
    return ply_vertex, ply_face


def construct_graph(vertex, face):
    n, m = vertex.shape[0], face.shape[0]
    g = ig.Graph(n)
    g.vs['index'] = [i for i in range(n)] # igraph subgraph 对节点重新编号需要记录绝对序号
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
        segment = g.get_shortest_paths(ROI_vertices[i], ROI_vertices[(i+1)%n])[0]
        border += segment[0:-1]
    return border


def get_subgraph_vertices(g, vertex, border = None):
    g1 = g.to_networkx()
    g1.remove_nodes_from(border)
    return list(nx.node_connected_component(g1, vertex))
