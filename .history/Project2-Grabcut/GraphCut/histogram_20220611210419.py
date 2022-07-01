import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"

if __name__ == "__main__":
    src = cv.imread(PATH+'pic.jpg', cv.IMREAD_GRAYSCALE)
    # plt.hist(src.ravel(), 16)
    # plt.hist(src.flatten(), 256)

    # ret_hist = cv.calcHist([src], [0], None, [256], [0, 255])
    # plt.plot(ret_hist)

    # plt.show()
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()