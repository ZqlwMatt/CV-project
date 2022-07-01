import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"

if __name__ == "__main__":
    src = cv.imread(PATH+'lena.jpg', cv.IMREAD_GRAYSCALE)
    
    # cv2.imwrite(PATH+'test.jpg', src)