import cv2 as cv
import numpy as np

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"


class Painter():
    def __init__(self, img):
        self.src = img.copy()
        self.img = img.copy()

        self.size = 4
        self.status = 1
        self.mode = 0
        self.windowname = "Paint"

    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            if self.mode == 0:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1) # 绿色填充 object
            else:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (255, 0, 0), -1) # 蓝色填充 background
            cv.imshow(self.img)
        elif event == cv.EVENT_LBUTTONUP:
            self.status = 0

    def _paint(self):
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onmouse)

        while(1):
            cv2.imshow(self.windowname, self.img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                self.mode = 1

class Picdata():
    def __init__(self):
        self.hist = None


    def calcPicHistogram():
        self.hist = cv.calcHist([src_gray], [0], None, [256], [0, 255])



if __name__ == "__main__":
    src_gray = cv.imread(PATH+'pic.jpeg', cv.IMREAD_GRAYSCALE)
    src = cv.imread(PATH+'pic.jpeg')

