import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import igraph as ig
from plyfile import PlyData
from utils import *
from solver import *

g = ig.Graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

