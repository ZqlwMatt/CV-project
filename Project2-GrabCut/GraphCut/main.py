import cv2 as cv
import numpy as np
from GraphPainter import *

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"

if __name__ == "__main__":
    PATH = input("Please input the path of your working directory: ")
    print("Launch: GraphCut")
    Painter = GraphPainter(PATH+'hat.jpg')
    res = Painter.paint()
    cv.imwrite(PATH+'result.png', res)