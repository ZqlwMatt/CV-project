import cv2 as cv
import numpy as np

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"


class Picdata():
    def __init__(self) -> None:
        self.ret_hist = None
    def calcPicHistogram():
        ret_hist = cv.calcHist([src_gray], [0], None, [256], [0, 255])



if __name__ == "__main__":
    src_gray = cv.imread(PATH+'pic.jpeg', cv.IMREAD_GRAYSCALE)
    src = cv.imread(PATH+'pic.jpeg')

