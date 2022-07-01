import cv2 as cv
import numpy as np

class Painter():
    def __init__(self, image_path):
        self.src = cv.imread(image_path)
        self.img = self.src.copy()

        self.size = 4
        self.status = 1   # Mouse status
        self.mode = 0     # Paint mode
        self.windowname = "Paint"

    def _onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.status = 1
        elif (self.status == 1) & (event == cv.EVENT_MOUSEMOVE):
            if self.mode == 0:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1) # 绿色填充 object
            else:
                cv.rectangle(self.img, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 0, 255), -1) # 红色填充 background
            cv.imshow(self.img)
        elif event == cv.EVENT_LBUTTONUP:
            self.status = 0

    def _paint(self):
        cv.namedWindow(self.windowname)
        cv.setMouseCallback(self.windowname, self._onmouse)

        while(1):
            cv2.imshow(self.windowname, self.img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.mode ^= 1
            elif key == ord('c'):
                
