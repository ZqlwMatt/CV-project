import cv2
import numpy as np

PATH = "/Users/zqlwmatt/project/CV-project/Project2-Grabcut/GraphCut/"

if __name__ == "__main__":
    src = cv2.imread('./lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(PATH+'test.jpg', src)