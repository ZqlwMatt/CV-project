import cv2 as cv
import numpy as np
import maxflow
from sklearn.cluster import KMeans

k = 5
inf = 450

class GMM:
    def __init__(self):
        self.weight  = np.asarray([0. for i in range(k)])                                          # Weight of each component
        self.mean    = np.asarray([[0., 0., 0.] for i in range(k)])                                # Means of each component
        self.cov     = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])  # Covs of each component
        self.cov_inv = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])
        self.cov_det = np.asarray([0. for i in range(k)])
        self.count   = np.asarray([0 for i in range(k)])                                           # Count of pixels in each components
        self.total_count = 0                                                                       # The total number of pixels in the GMM

        self.sum     = np.asarray([[0., 0., 0.] for i in range(k)])
        self.prod    = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])


    def Pr(self, ci, color):
        """
        -> The probability of color belonging to GMM[ci].
           color: (n, ) array
        """
        res = 0.
        if self.weight[ci] > 0:
            diff = np.asarray([color - self.mean[ci]])
            mult = np.dot(self.cov_inv[ci], np.transpose(diff))
            mult = np.dot(diff, mult)[0][0]
            res = np.exp(-0.5 * mult) / np.sqrt(self.cov_det[ci]) / np.sqrt(2 * np.pi)
        return res


    def totPr(self, color):
        """
        -> The probability of color belonging to background / foreground.
           color: (n, ) array
        """
        return np.sum([self.weight[ci] * self.Pr(ci, color) for ci in range(k)])
    

    def whichComponent(self, color):
        """
        -> The most likely GMM component.
        """
        prob = np.asarray([self.Pr(ci, color) for ci in range(k)])
        return prob.argmax(0)


    def addSample(self, ci, color):
        """
        -> Add a sample to train the GMM.
           color: (n, )
        """
        self.sum[ci] += color
        color = color.reshape(-1, 1)
        self.prod[ci] += np.dot(color, np.transpose(color))
        self.count[ci] += 1
        self.total_count += 1



    def clear(self):
        """
        -> initialize the GMM learning parameters.
        """
        # self.weight  = np.asarray([0. for i in range(k)])
        self.count   = np.asarray([0 for i in range(k)])
        self.total_count = 0

        self.sum     = np.asarray([[0., 0., 0.] for i in range(k)])
        self.prod    = np.asarray([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]] for i in range(k)])


    def Learning(self):
        """
        -> learn the Gussian Mixture Model.
        """
        variance = 0.01
        for ci in range(k):
            n = self.count[ci]
            if n == 0:
                self.weight[ci] = 0
            else:
                self.weight[ci] = n / self.total_count
                self.mean[ci] = self.sum[ci] / n
                mean_tmp = self.mean[ci].reshape(-1, 1)
                self.cov[ci] = self.prod[ci] / n - np.dot(mean_tmp, np.transpose(mean_tmp))
                self.cov_det[ci] = np.linalg.det(self.cov[ci])
                # 添加白噪声防止矩阵退化 (avoid singular matrix)
                while self.cov_det[ci] <= 0:
                    self.cov[ci] += np.diag([variance, variance, variance])
                    self.cov_det[ci] = np.linalg.det(self.cov[ci])
                self.cov_inv[ci] = np.linalg.inv(self.cov[ci])


class Solver:
    def __init__(self, img):
        self.src = img.copy()
        self.img = np.asarray(img, dtype=np.float64)     # 像素值运算需要类型转换
                                                         # unit8 类型：numpy 0-255 循环 | cv 溢出封顶
        self.h, self.w = img.shape[:2]
        self.size = self.h * self.w                      # picture / node size
        self.mask = np.zeros_like(self.img, dtype=bool)  # h * w
        self.graph = np.zeros((self.h, self.w))          # mask
        self.components = np.zeros((self.h, self.w), dtype=np.int64)

        self.foreSamples = []
        self.backSamples = []
        self.foreGMM = GMM()
        self.backGMM = GMM()

        self.gamma = 50
        self.beta = 0.


    def initGMM(self, rect, iternum = 200):
        lx, ly = rect[0], rect[1]
        rx, ry = rect[2], rect[3]

        for y in range(self.h):
            for x in range(self.w):
                idx = self.graph[y, x]
                if idx == 2:
                    self.backSamples.append(self.img[y, x])
                elif idx == 3:
                    self.foreSamples.append(self.img[y, x])
                elif (y >= ly) & (y <= ry) & (x >= lx) & (x <= rx):
                    self.foreSamples.append(self.img[y, x])
                    self.graph[y, x] = 1    # mark prob_foreground
                else:
                    self.backSamples.append(self.img[y, x])
                    self.graph[y, x] = 0    # mark prob_background

        self.foreSamples = np.asarray(self.foreSamples)
        self.backSamples = np.asarray(self.backSamples)
        """ scikit-learn package
        cluster_centers_: the coordinates of cluster centers.
        labels_: the sample labels.
        n_iter_: the itertion number.
        """
        KMeans_model_fore = KMeans(n_clusters = k, max_iter = iternum)
        fore_res = KMeans_model_fore.fit(self.foreSamples)
        KMeans_model_back = KMeans(n_clusters = k, max_iter = iternum)
        back_res = KMeans_model_back.fit(self.backSamples)

        for i in range(len(self.foreSamples)):
            self.foreGMM.addSample(fore_res.labels_[i], self.foreSamples[i])
        for i in range(len(self.backSamples)):
            self.backGMM.addSample(back_res.labels_[i], self.backSamples[i])

        self.foreGMM.Learning()
        self.backGMM.Learning()

    
    def calcBeta(self):
        beta = 0.
        for (y, x), idx in np.ndenumerate(self.graph):
            if x > 0:
                diff = self.img[y, x] - self.img[y, x-1]
                beta += np.sum(np.square(diff))
            if (x > 0) & (y > 0):
                diff = self.img[y, x] - self.img[y-1, x-1]
                beta += np.sum(np.square(diff))
            if y > 0:
                diff = self.img[y, x] - self.img[y-1, x]
                beta += np.sum(np.square(diff))
            if (y > 0) & (x < self.w-1):
                diff = self.img[y, x] - self.img[y-1, x+1]
                beta += np.sum(np.square(diff))
        
        beta = 1. / (2 * beta / (4 * self.size - 3 * self.w - 3 * self.h + 2))

        return beta


    def assignComponents(self):
        for (y, x), idx in np.ndenumerate(self.graph):
            if (idx == 1) | (idx == 3):
                self.components[y, x] = self.foreGMM.whichComponent(self.img[y, x])
            else:
                self.components[y, x] = self.backGMM.whichComponent(self.img[y, x])


    def learnGMM(self):
        self.foreGMM.clear()
        self.backGMM.clear()

        for (y, x), idx in np.ndenumerate(self.graph):
            color = self.img[y, x]
            k = self.components[y, x]
            if (idx == 1) | (idx == 3):
                self.foreGMM.addSample(k, color)
            else:
                self.backGMM.addSample(k, color)
        
        self.foreGMM.Learning()
        self.backGMM.Learning()
        print('GMM learning finished...')
        

    def segmentGraph(self):
        g = maxflow.Graph[float](self.size, (4 * self.size - 3 * self.h - 3 * self.w + 2) * 2)
        nodes = g.add_nodes(self.size)

        beta = self.beta
        gamma = self.gamma
        gammaDivSqrt2 = gamma / np.sqrt(2.)
        for (y, x), idx in np.ndenumerate(self.graph):
            k = y * self.w + x
            color = self.img[y, x]
            if x > 0:
                diff = color - self.img[y, x-1]
                w = gamma * np.exp(- beta * np.sum(np.square(diff)))
                g.add_edge(k, k-1, w, w)
            if (x > 0) & (y > 0):
                diff = color - self.img[y-1, x-1]
                w = gammaDivSqrt2 * np.exp(- beta * np.sum(np.square(diff)))
                g.add_edge(k, k-1-self.w, w, w)
            if y > 0:
                diff = color - self.img[y-1, x]
                w = gamma * np.exp(- beta * np.sum(np.square(diff)))
                g.add_edge(k, k-self.w, w, w)
            if (y > 0) & (x < self.w-1):
                diff = color - self.img[y-1, x+1]
                w = gammaDivSqrt2 * np.exp(- beta * np.sum(np.square(diff)))
                g.add_edge(k, k-self.w+1, w, w)
        
        print('n_link is connected...')

        for (y, x), idx in np.ndenumerate(self.graph):
            k = y * self.w + x
            if self.graph[y, x] == 2:
                g.add_tedge(k, inf, 0)
            elif self.graph[y, x] == 3:
                g.add_tedge(k, 0, inf)
            else:
                color = self.img[y, x]
                w_f = self.foreGMM.totPr(color)
                w_f = - np.log(w_f) if w_f > 0 else inf
                w_b = self.backGMM.totPr(color)
                w_b = - np.log(w_b) if w_b > 0 else inf
                g.add_tedge(k, w_f, w_b)
        
        print('t_link is connected...')

        g.maxflow()
        print('minsegment finished...')

        for (y, x), idx in np.ndenumerate(self.graph):
            if idx < 2:
                self.graph[y, x] = g.get_segment(y * self.w + x)
    

    def grabCut(self, rect, iternum = 3):
        self.initGMM(rect)
        self.beta = self.calcBeta()
        print("beta = " + str(self.beta))

        for _ in range(iternum):
            print(_)
            self.assignComponents()
            self.learnGMM()
            self.segmentGraph()

        for (y, x), idx in np.ndenumerate(self.graph):
            if (idx == 1) | (idx == 3):
                self.mask[y, x] = (True, True, True)

        result = np.zeros_like(self.src)
        np.copyto(result, self.src, where=self.mask)
        return result


    """
    opencv 提供的方法
    """
    def _grabCut(self, rect):
        img = self.img.copy()
        mask = np.zeros(img.shape[:2], np.uint8)
        lx, ly = rect[0], rect[1]
        rx, ry = rect[2], rect[3]

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        rect = (lx, ly, rx, ry)

        # for x in range(lx, rx):
        #     for y in range(ly, ry):
        #         mask[y, x] = (1, 1, 1)
        
        mask.astype('uint8')
        cv.grabCut(self.img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        return img

