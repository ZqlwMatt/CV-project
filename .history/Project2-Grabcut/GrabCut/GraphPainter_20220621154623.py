import cv2 as cv
import numpy as np
import maxflow
from solver import *

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
        res = np.zeros()
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
                # 求 解 部 分
                res = self.solver.graphCut((self.lx, self.ly), (self.rx, self.ry))
                break
            elif key == 27:
                cv.destroyAllWindows()
                exit()
                
        cv.destroyAllWindows()
        return res

