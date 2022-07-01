import cv2 as cv
import numpy as np
import maxflow
from sklearn.cluster import KMeans

k = 5
class GMM():
    def __init__(self):
        self.weights = np.asarray([0. for i in range(k)])                                          # Weight of each component
        self.means = np.asarray([[0., 0., 0.] for i in range(k)])                                  # Means of each component
        self.covs = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])     # Covs of each component
        self.cov_inv = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])
        self.cov_det = np.asarray([0. for i in range(k)])
        self.counts = np.asarray([0. for i in range(k)])                                           # Count of pixels in each components
        self.total_count = 0                                                                       # The total number of pixels in the GMM
        
        # The following two parameters are assistant parameters for counting pixels and calc. pars.
        self.sums = np.asarray([[0., 0., 0.] for i in range(k)])
        self.prods = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])


    def Pr(self, ci, color):
        res = 0
        if self.weights > 0:
            # if cov_det[i] < 0   GG
            diff = color.copy() - self.means
            diff.shape = (-1, 1)
            res = 1. / np.sqrt(self.cov_det[ci]) * np.exp(-0.5 * np.dot(np.dot(np.transpose(diff), self.cov_inv), diff))
            # 常数项直接丢掉
        return res


    def whichComponent(self, color):
        pos = 0
        Max = 0
        for ci in range(k):
            p = self.Pr(ci, color)
            if p > Max:
                pos = ci
                Max = p
        
        return pos


    def addSample(self, ci, w):
        self.sums[ci] += w
        w.shape = (-1, 1)
        self.prods[ci] += np.dot(w, np.transpose(w))
        self.counts[ci] += 1
        self.total_count += 1


    def clear(self):
        self.counts = 0
        self.total_count = 0
        self.sums = np.asarray([[0., 0., 0.] for i in range(k)])
        self.prods = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])


    def Learning(self):
        # Prepare GMM Matries
        variance = 0.01
        for ci in range(k):
            n = self.counts[ci]
            if n == 0:
                self.weights[ci] = 0
            else:
                self.weights[ci] = n / self.total_count
                self.means[ci] = self.sums / n
                mean = self.means[ci].copy()
                mean.shape = (-1, 1)
                self.covs[ci] = self.prods[ci] / n - np.dot(mean, np.transpose(mean))
                self.cov_det[ci] = np.linalg.det(self.covs[ci])
                # 添加白噪声防止矩阵退化 (avoid singular matrix)
                while self.cov_det[ci] <= 0:
                    self.covs[ci] += np.diag([variance, variance, variance])
                    self.cov_det[ci] = np.linalg.det(self.covs[ci])
                self.cov_inv[ci] = np.linalg.inv(self.covs[ci])



class Solver():
    alpha = 50.
    inf = 1000000
    inf_ = 100
    iternum = 200
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
        """初始化 Guassin Mixture Model : 用 KMeans 学习 label.
        """
        self.foreGMM.clear()
        self.backGMM.clear()

        lx, ly = min(p1[0], p2[0]), min(p1[1], p2[1])
        rx, ry = max(p1[0], p2[0]), max(p1[1], p2[1])

        for y in range(self.h):
            for x in range(self.w):
                if (y >= ly) & (y <= ry) & (x >= lx) & (x <= rx):
                    self.foreSamples.append(self.img[y, x])
                    self.graph[y, x] = 1
                else:
                    self.backSamples.append(self.img[y, x])
                    self.graph[y, x] = 0

        """ scikit-learn package
        cluster_centers_: 聚类中心的坐标。
        如果算法还未完全收敛就停止，则将与labels_不一致。
        labels_: 每个点的标签。
        inertia_: 样本到聚类中心的平方和。
        n_iter_: 迭代运行的次数。
        """
        KMeans_model = KMeans(n_clusters = k, max_iter = self.iternum)
        fore_res = KMeans_model(self.foreSamples)
        back_res = KMeans_model(self.backSamples)

        for i in range(len(self.foreSamples)):
            self.foreGMM.addSample(fore_res.label_[i], self.foreSamples[i])
        for i in range(len(self.backSamples)):
            self.backGMM.addSample(back_res.label_[i], self.backSamples[i])
        
        self.foreGMM.Learning()
        self.backGMM.Learning()

    
    def learnGMM(self):
        """GMM 参数学习
        """
        for (y, x), idx in np.ndenumerate(self.graph):
            color = self.img[y, x]
            if idx == 1:
                self.foreGMM.addSample(self.foreGMM.whichComponent(color), color)
            else:
                self.backGMM.addSample(self.backGMM.whichComponent(color), color)
        
        self.foreGMM.Leanring()
        self.foreGMM.Leanring()


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
        

    def n_link(self, beta, gamma):
        gammaDivSqrt2 = gamma / np.sqrt(2.)
        for (y, x), idx in np.enumerate(self.graph):
            k = y * self.w + x
            if x >= 1:
                diff = self.img[y, x] - self.img[y, x-1]
                self.g.add_edge(k, k-1, gamma * np.exp(- beta * np.power(diff, 2)))
            else:
                self.g.add_edge(k, k-1, 0, 0)
            if (x >= 1) & (y >= 1):
                diff = self.img[y, x] - self.img[y-1, x-1]
                

        # for (y, x), idx in np.ndenumerate(self.graph):
        #     k = y * self.w + x
        #     if x != self.w-1:
        #         w = 1. / (1. + np.sum(np.power(self.img[y, x+1] - self.img[y, x], 2) / 2))
        #         self.g.add_edge(k, k+1, w, w)
        #     if y != self.h-1:
        #         w = 1. / (1. + np.sum(np.power(self.img[y+1, x] - self.img[y, x], 2) / 2))
        #         self.g.add_edge(k+self.w, k, w, w)



    def graphCut(self, p1, p2, iter = 10):
        self.initGMM(p1, p2)
        self.n_link()

        nodes = self.g.add_nodes(self.size)
        for _ in range(iter):
            """ 循环
            GMM 模型优化
            建立 t-link
            跑图 resampling
            """
            self.learnGMM()
            self.constructGraph()

        # self.t_link()
        # self.n_link()
        self.g.maxflow()
        for i in nodes:
            if self.g.get_segment(i) == 1:
                self.mask[i // self.w, i % self.w] = (True, True, True)

        result = np.zeros_like(self.img)
        np.copyto(result, self.img, where=self.mask)
        return result
        