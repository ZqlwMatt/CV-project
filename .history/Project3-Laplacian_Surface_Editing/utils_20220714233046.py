import igraph as ig
import scipy.sparse

def numpyToPly(vertex, face):
    ply_vertex = np.empty(vertex.shape[0], dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_face = np.empty(face.shape[0], dtype = [('vertex_indices', 'i4', (3,))])
    ply_vertex['x'] = n_vertex[:, 0]
    ply_vertex['y'] = n_vertex[:, 1]
    ply_vertex['z'] = n_vertex[:, 2]
    ply_face['vertex_indices'] = n_face
    ply_vertex = PlyElement.describe(ply_vertex, 'vertex')
    ply_face = PlyElement.describe(ply_face, 'face')
