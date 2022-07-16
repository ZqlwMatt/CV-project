import numpy as np
import igraph as ig
from plyfile import PlyData

class NumpyStudy:
    def verticalStackArray(self):
        array1 = np.array([1, 2, 3])
        array2 = np.array([2, 3, 4])
        array = np.vstack((array1, array2))
        print("数组array的值为: ")
        print(array)


if __name__ == "__main__":
    main = NumpyStudy()
    main.verticalStackArray()
# igraph 的 subgraph 会重新编号