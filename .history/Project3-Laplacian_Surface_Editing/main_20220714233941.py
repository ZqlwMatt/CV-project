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
    vertex, face = plyToNumpy(plydata)

    

    ply_vertex, ply_face = numpyToPly(vertex, face)
    PlyData([ply_vertex, ply_face], text=True).write(PATH+'test.ply')