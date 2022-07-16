import numpy as np
import networkx as nx
import igraph
import scipy.sparse
import scipy.sparse.linalg
from plyfile import PlyData


def convert_mesh_to_graph(plydata):
    g = igraph.Graph()
    vertices_raw = plydata.elements[0].data
    edges_raw = plydata.elements[1].data
    edges = set()
    for edge_attrs in edges_raw:
        edge_attrs[0] = sorted(edge_attrs[0])
        edges.add((edge_attrs[0][0], edge_attrs[0][1]))
        edges.add((edge_attrs[0][0], edge_attrs[0][2]))
        edges.add((edge_attrs[0][1], edge_attrs[0][2]))
    for i in range(len(vertices_raw)):
        g.add_vertex(i, index=i)
    g.add_edges(edges)
    return g


def get_border_vertices(g, vertices):
    boundary = []
    for i in range(len(vertices)):
        vertex1 = vertices[i]
        vertex2 = vertices[(i + 1) % len(vertices)]
        boundary.extend((g.get_all_shortest_paths(vertex1, vertex2))[0][0:-1])
    return boundary


def get_morph_vertices(g, border, handle):
    sub_g = g.to_networkx().copy()
    sub_g.remove_nodes_from(border)
    return list(nx.node_connected_component(sub_g, handle))


def calc_laplacian_matrix(g):
    # Lrw = I - (D^-1)A = (D^-1)L
    D = np.diag(g.degree(list(range(morph_num))))
    L = np.array(g.laplacian())
    return np.linalg.inv(D).dot(L)


def get_neighbors(g, vertex):
    neighbors_local = g.neighbors(vertex)
    return neighbors_local


ply_data = PlyData.read('./meshs/bun_zipper.ply')
# ply_data = PlyData.read('./meshs/dragon_vrip.ply')
total_graph = convert_mesh_to_graph(ply_data)
total_vertex_pos = np.vstack(list(vertex_attrs)[0:3] for vertex_attrs in ply_data.elements[0].data)


handle_vertices = {
    14651: [-0.0123491, 0.153911, -0.0264322],    # bunny
    # 314362: [-0.073, 0.17983, -0.1],              # dragon
}
ROI_vertices = [15692, 7357, 9877, 28992]     # bunny
# ROI_vertices = [338016, 201222, 74997, 216233]  # dragon
border_vertices = get_border_vertices(total_graph, ROI_vertices)
# print(border_vertices)
morph_vertices = get_morph_vertices(total_graph, border_vertices, list(handle_vertices.keys())[0])
morph_vertices = border_vertices + morph_vertices
# print(morph_vertices)

sub_graph = total_graph.subgraph(morph_vertices)
morph_num = len(morph_vertices)
fixed_num = len(handle_vertices) + len(ROI_vertices)
total_to_sub = {}
sub_to_total = {}
for i in range(morph_num):
    vertex = sub_graph.vs[i]
    total_to_sub[vertex['index']] = i
    sub_to_total[i] = vertex['index']

L = calc_laplacian_matrix(sub_graph)
# print(np.linalg.matrix_rank(L))
V = np.vstack(total_vertex_pos[sub_to_total[i]] for i in range(morph_num))
Delta = L.dot(V)
# print(Delta)

L_prime = np.zeros([3 * (morph_num + fixed_num), 3 * morph_num])
L_prime[0 * morph_num:1 * morph_num, 0 * morph_num:1 * morph_num] = L
L_prime[1 * morph_num:2 * morph_num, 1 * morph_num:2 * morph_num] = L
L_prime[2 * morph_num:3 * morph_num, 2 * morph_num:3 * morph_num] = L
for i in range(morph_num):
    cur_vertices = get_neighbors(sub_graph, i)
    cur_vertices.append(i)
    Ai = np.zeros([3 * len(cur_vertices), 7])
    for j in range(len(cur_vertices)):
        pos = V[cur_vertices[j]]
        Ai[j] = [pos[0], 0, pos[2], -pos[1], 1, 0, 0]
        Ai[j + len(cur_vertices)] = [pos[1], -pos[2], 0, pos[0], 0, 1, 0]
        Ai[j + 2 * len(cur_vertices)] = [pos[2], pos[1], -pos[0], 0, 0, 0, 1]

    Delta_i_x = Delta[i, 0]
    Delta_i_y = Delta[i, 1]
    Delta_i_z = Delta[i, 2]

    # (si, hi, ti)_T = (Ai_T*Ai)^(-1)*Ai_T
    sht = np.linalg.inv(Ai.T.dot(Ai)).dot(Ai.T)
    si = sht[0]
    hi = sht[1:4]
    ti = sht[4:7]

    T_delta = [si * Delta_i_x - hi[2] * Delta_i_y + hi[1] * Delta_i_z,
               hi[2] * Delta_i_x + si * Delta_i_y - hi[0] * Delta_i_z,
               -hi[1] * Delta_i_x + hi[0] * Delta_i_y + si * Delta_i_z]

    cur_vertices = np.array(cur_vertices)
    row_indices = np.hstack([cur_vertices, cur_vertices + morph_num, cur_vertices + 2 * morph_num])
    L_prime[i + 0 * morph_num, row_indices] = (-1) * L_prime[i + 0 * morph_num, row_indices] + T_delta[0]
    L_prime[i + 1 * morph_num, row_indices] = (-1) * L_prime[i + 1 * morph_num, row_indices] + T_delta[1]
    L_prime[i + 2 * morph_num, row_indices] = (-1) * L_prime[i + 2 * morph_num, row_indices] + T_delta[2]

b = np.array([])
for i in range(len(handle_vertices)):
    handle_vertex = list(handle_vertices.keys())[i]
    b = np.append(b, handle_vertices[handle_vertex])
    handle_vertex_index_sub = total_to_sub[handle_vertex]
    L_prime[3 * morph_num + i * 3 + 0, handle_vertex_index_sub + 0 * morph_num] = 1
    L_prime[3 * morph_num + i * 3 + 1, handle_vertex_index_sub + 1 * morph_num] = 1
    L_prime[3 * morph_num + i * 3 + 2, handle_vertex_index_sub + 2 * morph_num] = 1
for i in range(len(ROI_vertices)):
    b = np.append(b, V[total_to_sub[ROI_vertices[i]]])
    ROI_vertex_index_sub = total_to_sub[ROI_vertices[i]]
    L_prime[3 * (morph_num + len(handle_vertices)) + i * 3 + 0, ROI_vertex_index_sub + 0 * morph_num] = 1
    L_prime[3 * (morph_num + len(handle_vertices)) + i * 3 + 1, ROI_vertex_index_sub + 1 * morph_num] = 1
    L_prime[3 * (morph_num + len(handle_vertices)) + i * 3 + 2, ROI_vertex_index_sub + 2 * morph_num] = 1

b = np.hstack([np.zeros(3 * morph_num), b])
spA = scipy.sparse.coo_matrix(L_prime)
new_V = scipy.sparse.linalg.lsqr(spA, b)[0]
# print(new_V)

for i in range(morph_num):
    total_vertex_pos[sub_to_total[i]] = [new_V[i], new_V[i + morph_num], new_V[i + 2 * morph_num]]
print(total_vertex_pos)

for i in range(total_vertex_pos.shape[0]):
    ply_data.elements[0].data[i] = tuple(total_vertex_pos[i]) + tuple(ply_data.elements[0].data[i])[3:]
ply_data.write('./edit_meshs/bun_zipper_edited.ply')
# ply_data.write('./edit_meshs/dragon_vrip_edited.ply')
