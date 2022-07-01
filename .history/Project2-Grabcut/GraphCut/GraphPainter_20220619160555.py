import cv2 as cv
import numpy as np

class Solver():
    def __init__(self, image_path):

        

class GraphPainter():
    def __init__(self, image_path):
        self.src = cv.imread(image_path)
        self.img = self.src.copy()

        self.size = 1
        self.status = 1   # Mouse status
        self.mode = 0     # Paint mode
        self.windowname = "123"

        self.Solver = 
        self.fore_seeds = []
        self.back_seeds = []


    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            if self.mode == 0:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1) # 绿色填充 object
                if not self.fore_seeds.__contains__((x, y)):
                    self.fore_seeds.append((x, y))
            else:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 0, 255), -1) # 红色填充 background
                if not self.back_seeds.__contains__((x, y)):
                    self.back_seeds.append((x, y))
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
                self.fore_seeds = []
                self.back_seeds = []
            elif key == ord('g'):
                break
            elif key == 27:
                break
                
        cv.destroyAllWindows()
        return 

