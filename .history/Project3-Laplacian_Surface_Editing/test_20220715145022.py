import numpy as np
import igraph as ig
from plyfile import PlyData

g = ig.Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

g.vs['index'] = [i for i in range(4)]

g1 = g.induced_subgraph([1, 2])

print(g1.vs['index'])

# igraph 的 subgraph 会重新编号