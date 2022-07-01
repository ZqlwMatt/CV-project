import cv2 as cv
import numpy as np
import maxflow

class Solver():
    alpha = 50.
    inf = 1000000
    inf_ = 100
    def __init__(self, img):
        self.img = img.copy()
        self.h, self.w = img.shape[:2]
        self.mask = np.zeros_like(self.img, dtype=bool) # h * w

        self.fore_seeds = []
        self.back_seeds = []
        self.graph = (-1) * np.ones((self.h, self.w))  # draft
        print(self.graph.shape)
        self.size = self.h * self.w
        
        self.g = maxflow.Graph[float](self.size, (2 * self.size - self.h - self.w) * 2)


    def sampling(self, p1, p2):
        lx, ly = min(p1[0], p2[0]), min(p1[1], p2[1])
        rx, ry = max(p1[0], p2[0]), max(p1[1], p2[1])

        for (y, x) in np.ndenumerate()
        for x in range(lx, ly+1):
            for y in range(ly, ry+1):
                self.fore_seed.append((x, y))
        

    def init_grayhist(self):
        # 灰度直方图归一化
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        cv.normalize(self.gray_img, self.gray_img, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        self.gray_hist = cv.calcHist([self.gray_img], [0], None, [256], [0, 256])
        self.back_gray_hist = np.zeros(256)
        self.fore_gray_hist = np.zeros(256)

        for (y, x), idx in np.ndenumerate(self.graph):
            if idx == 0:
                self.back_gray_hist[self.gray_img[y, x]] += 1
            elif idx == 1:
                self.fore_gray_hist[self.gray_img[y, x]] += 1


    def calc_gray(self, y, x):
        num = self.gray_img[y, x] # 灰度值
        p1, p2 = self.back_gray_hist[num], self.fore_gray_hist[num]
        if p1+p2 == 0:
            return self.inf_, self.inf_
        if p1 == 0:
            return self.inf_, -np.log(p2 / (p1+p2))
        if p2 == 0:
            return -np.log(p1 / (p1+p2)), self.inf_
        return -np.log(p1 / (p1+p2)), -np.log(p2 / (p1+p2))


    def t_link(self):
        self.init_grayhist()
        for (x, y) in self.fore_seeds:
            self.graph[y, x] = 1
        for (x, y) in self.back_seeds:
            self.graph[y, x] = 0
        for (y, x), idx in np.ndenumerate(self.graph):
            k = y * self.w + x
            if idx == 0:
                self.g.add_tedge(k, self.inf, 0)
            elif idx == 1:
                self.g.add_tedge(k, 0, self.inf)
            else:
                w = self.calc_gray(y, x)
                self.g.add_tedge(k, self.alpha * w[0], self.alpha * w[1])
        

    def n_link(self):
        for (y, x), idx in np.ndenumerate(self.graph):
            k = y * self.w + x
            if x != self.w-1:
                w = 1. / (1. + np.sum(np.power(self.img[y, x+1] - self.img[y, x], 2) / 2))
                self.g.add_edge(k, k+1, w, w)
            if y != self.h-1:
                w = 1. / (1. + np.sum(np.power(self.img[y+1, x] - self.img[y, x], 2) / 2))
                self.g.add_edge(k+self.w, k, w, w)


    def graphCut(self):
        nodes = self.g.add_nodes(self.size)
        self.sampling()
        self.t_link()
        self.n_link()
        self.g.maxflow()
        for i in nodes:
            if self.g.get_segment(i) == 1:
                self.mask[i // self.w, i % self.w] = (True, True, True)

        result = np.zeros_like(self.img)
        np.copyto(result, self.img, where=self.mask)
        return result
        

class GraphPainter():
    def __init__(self, image_path):
        self.src = cv.imread(image_path)
        self.img = self.src.copy()

        self.size = 1
        self.status = 0   # Mouse status
        self.mode = 0     # Paint mode
        self.lx, self.ly = 0, 0 # start point
        self.rx, self.ry = 0, 0
        self.windowname = "123"

        self.solver = Solver(self.src)


    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
            self.lx, self.ly = x, y
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            cv.rectangle(self.img, (self.x, self.y), (x, y), (0, 0, 255), self.size)
            self.rx, self.ry = x, y
            # if self.mode == 0:
            #     cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1) # 绿色填充 object
            #     if not self.solver.fore_seeds.__contains__((x, y)):
            #         self.solver.fore_seeds.append((x, y))
            # else:
            #     cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 0, 255), -1) # 红色填充 background
            #     if not self.solver.back_seeds.__contains__((x, y)):
            #         self.solver.back_seeds.append((x, y))
        elif event == cv.EVENT_LBUTTONUP:
            self.status = 0


    def paint(self):
        res = None
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onMouse)

        while(1):
            cv.imshow(self.windowname, self.img)
            key = cv.waitKey(1) & 0xFF

            if key == ord('t'):
                self.mode ^= 1
            elif key == ord('c'):
                self.img = self.src.copy()
            elif key == ord('g'):
                res = self.solver.graphCut((self.lx, self.ly), (self.rx, self.ry))
                break
            elif key == 27:
                cv.destroyAllWindows()
                exit()
                
        cv.destroyAllWindows()
        return res

