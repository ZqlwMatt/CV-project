import cv2 as cv
import numpy as np

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"


class Painter():
    def __init__(self, img):
        self.src = img.copy()
        self.img = img.copy()
        self.windowname = "Paint"

    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            

    def _paint(self):
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onmouse)


class Picdata():
    def __init__(self):
        self.hist = None


    def calcPicHistogram():
        self.hist = cv.calcHist([src_gray], [0], None, [256], [0, 255])



if __name__ == "__main__":
    src_gray = cv.imread(PATH+'pic.jpeg', cv.IMREAD_GRAYSCALE)
    src = cv.imread(PATH+'pic.jpeg')

