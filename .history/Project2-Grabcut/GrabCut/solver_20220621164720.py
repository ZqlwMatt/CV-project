import cv2 as cv
import numpy as np
import maxflow
from sklearn.cluster import KMeans

class GMM():
    def __init__(self):
        self.model = None


class Solver():
    alpha = 50.
    inf = 1000000
    inf_ = 100
    k = 5
    iternum = 20
    def __init__(self, img):
        self.img = img.copy()
        self.h, self.w = img.shape[:2]
        self.size = self.h * self.w
        self.mask = np.zeros_like(self.img, dtype=bool) # h * w
        self.graph = (-1) * np.ones((self.h, self.w))   # 草稿纸
        print(self.graph.shape)

        self.foreSamples = []
        self.backSamples = []
        self.foreGMM = GMM()
        self.backGMM = GMM()
        
        self.g = maxflow.Graph[float](self.size, (2 * self.size - self.h - self.w) * 2)


    def initGMM(self, p1, p2):
        lx, ly = min(p1[0], p2[0]), min(p1[1], p2[1])
        rx, ry = max(p1[0], p2[0]), max(p1[1], p2[1])

        for (y, x), idx in np.ndenumerate(self.graph):
            if (y >= ly) & (y <= ry) & (x >= lx) & (x <= rx):
                self.foreSamples.append(self.img[y, x])
            else:
                self.backSamples.append(self.img[y, x])

        KMeans_model = KMeans(n_clusters = self.k, max_iter = self.iternum)
        fore_res = KMeans_model(self.foreSamples)
        back_res = KMeans_model(self.backSamples)
        """
        **cluster_centers_：**聚类中心的坐标。
        如果算法还未完全收敛就停止，则将与labels_不一致。
        **labels_：**每个点的标签。
        **inertia_：**样本到聚类中心的平方和。
        **n_iter_：**迭代运行的次数。
        """
        
    

    
    """
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
    """


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


    def graphCut(self, p1, p2, iter = 10):
        self.initGMM(p1, p2)

        nodes = self.g.add_nodes(self.size)
        for _ in range(iter):
            """
            前景背景 k-means 分类
            生成 GMM 模型
            建立 t-link, n-link
            跑图 resampling
            """
            sample_fore = kmeans(self.fore_seeds, self.k).run()
            sample_back = kmeans(self.back_seeds, self.k).run()

        # self.t_link()
        # self.n_link()
        self.g.maxflow()
        for i in nodes:
            if self.g.get_segment(i) == 1:
                self.mask[i // self.w, i % self.w] = (True, True, True)

        result = np.zeros_like(self.img)
        np.copyto(result, self.img, where=self.mask)
        return result
        