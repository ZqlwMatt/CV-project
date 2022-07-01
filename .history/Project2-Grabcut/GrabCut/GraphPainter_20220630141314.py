import cv2 as cv
import numpy as np
from solver import *

class GraphPainter():
    def __init__(self, image_path):
        self.src = cv.imread(image_path)
        self.img = self.src.copy()

        self.size = 4
        self.status = 0             # Mouse status
        self.mode = 1               # Paint mode : background (2) | foreground (3) |  prob_rect (1) (default)
        self.lx, self.ly = -1, -1   # start point
        self.rx, self.ry = -1, -1
        self.windowname = "input image"

        self.solver = Solver(self.src)


    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
            self.lx, self.ly = x, y
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            if self.mode == 1:
                self.img = self.src.copy()
                cv.rectangle(self.img, (self.lx, self.ly), (x, y), (255, 0, 0), self.size)
                self.rx, self.ry = x, y
            elif self.mode == 2:
                cv.rectangle(self.img, (x, y), (x, y), (255, 0, 0), self.size)
                self.solver.graph[y, x] = 2
            elif self.mode == 3:
                self.img[y, x] = (0, 255, 0)
                self.solver.graph[y, x] = 3
        elif event == cv.EVENT_LBUTTONUP:
            self.status = 0


    def paint(self):
        res = np.zeros_like(self.img)
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onMouse)

        while(1):
            cv.imshow(self.windowname, self.img)
            key = cv.waitKey(1) & 0xFF

            if key == ord('c'):
                self.img = self.src.copy()
            elif key == ord('g'):
                rect = (min(self.lx, self.rx), min(self.ly, self.ry),
                        max(self.lx, self.rx), max(self.ly, self.ry))
                res = self.solver.grabCut(rect)
                break
            elif key == ord('0'):
                print("Please mark the background pixels.")
                self.mode = 2
            elif key == ord('1'):
                print("Please mark the foreground pixels.")
                self.mode = 3
            elif key == 27:
                cv.destroyAllWindows()
                exit()
                
        cv.destroyAllWindows()
        return res

