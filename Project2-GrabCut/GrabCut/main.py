import cv2 as cv
import numpy as np
from GraphPainter import *

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GrabCut/"

if __name__ == "__main__":
    print("Launch: GraphCut")
    # PATH = input("Please input the path of working dirctory: ")
    Painter = GraphPainter(PATH+'boat.jpg')
    res = Painter.paint()
    cv.imwrite(PATH+'result.png', res)