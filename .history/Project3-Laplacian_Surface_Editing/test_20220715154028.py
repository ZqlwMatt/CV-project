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
————————————————
版权声明：本文为CSDN博主「勤奋的大熊猫」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/u011699626/article/details/117458115
# igraph 的 subgraph 会重新编号