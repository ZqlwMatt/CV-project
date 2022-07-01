import cv2 as cv
import numpy as np
import maxflow

class Solver():
    inf = 1000000
    def __init__(self, img):
        self.img = img.copy()
        self.mask = np.zeros(img.shape) # h * w

        self.fore_seeds = []
        self.back_seeds = []
        self.graph = (-1) * np.ones(img.shape)
        self.size = np.multiply(img.shape[:2])

        self.g = maxflow.Graph[float](self.size, 2 * self.size - np.sum(img.shape[:2]))


    def calc():

    def t_link(self):
        for (x, y) in self.fore_seeds:
            self.graph[y, x] = 1
        for (x, y) in self.back_seeds:
            self.graph[y, x] = 0
        for (y, x), w in self.ndnumerate(self.graph):
            if w == 0:
                self.add_tedge(y*)
            elif w == 1:

            else:



    def n_link(self):
        pass


    def graphCut(self):
        nodes = self.g.add_nodes(self.size)
        self.t_link()
        self.n_link()
        pass

        

class GraphPainter():
    def __init__(self, image_path):
        self.src = cv.imread(image_path)
        self.img = self.src.copy()

        self.size = 1
        self.status = 0   # Mouse status
        self.mode = 0     # Paint mode
        self.windowname = "123"

        self.solver = Solver(self.img)


    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            if self.mode == 0:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1) # 绿色填充 object
                if not self.solver.fore_seeds.__contains__((x, y)):
                    self.solver.fore_seeds.append((x, y))
            else:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 0, 255), -1) # 红色填充 background
                if not self.solver.back_seeds.__contains__((x, y)):
                    self.solver.back_seeds.append((x, y))
        elif event == cv.EVENT_LBUTTONUP:
            self.status = 0


    def paint(self):
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onMouse)

        while(1):
            cv.imshow(self.windowname, self.img)
            key = cv.waitKey(1) & 0xFF

            if key == ord('t'):
                self.mode ^= 1
            elif key == ord('c'):
                self.img = self.src.copy()
                self.solver.fore_seeds = []
                self.solver.back_seeds = []
            elif key == ord('g'):
                self.solver.graphCut()
                break
            elif key == 27:
                break
                
        cv.destroyAllWindows()
        return 

