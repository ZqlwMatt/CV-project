import cv2 as cv
import numpy as np
from GraphPainter import *

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"


if __name__ == "__main__":
    print("Launch: GraphCut")
    # PATH = input("Please input the path of working dirctory: ")
    Painter = GraphPainter(PATH+'hat.jpg')
    res = Painter.paint()
    if res != None:
        cv.imwrite()