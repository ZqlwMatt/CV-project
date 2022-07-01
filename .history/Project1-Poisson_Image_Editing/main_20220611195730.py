import cv2
import numpy as np
import scipy.signal
import scipy.linalg
import scipy.sparse
from mathmodule import *
# from os import path

PATH = "/Users/zqlwmatt/project/CV-project/Project1-Poisson_Image_Editing/"

class MaskPainter():
    # 依据 source 绘制一个 mask
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.mask = np.zeros(self.image.shape)
        self.image_copy = self.image.copy()
        self.mask_copy = self.mask.copy()

        self.status = 0 # 鼠标状态
        self.size = 4 # 方块大小
        self.windowname = "Select a region to blend. (s: save, r: reset, esc: exit)"


    def _onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: 
            self.status = 1
            # 检测左键按下
        elif (self.status == 1) & (event == cv2.EVENT_MOUSEMOVE):
            cv2.rectangle(self.image, (x-self.size, y-self.size), (x+self.size, y+self.size), (0, 255, 0), -1)     # 原图绿色填充
            cv2.rectangle(self.mask,  (x-self.size, y-self.size), (x+self.size, y+self.size), (255, 255, 255), -1) # mask记录填充区域
            cv2.imshow(self.windowname, self.image)
            # 实时移动绘制 mask
        elif event == cv2.EVENT_LBUTTONUP:
            self.status = 0
            # 检测左键释放


    def _paint(self):
        cv2.namedWindow(self.windowname)
        cv2.setMouseCallback(self.windowname, self._onMouse)

        while True:
            cv2.imshow(self.windowname, self.image)
            key = cv2.waitKey(1) & 0xFF
            # 检测按键
            if key == ord('s'):
                break
            elif key == ord('r'):
                self.image = self.image_copy
                self.mask = self.mask_copy
            elif key == 27:
                cv2.destroyAllWindows()
                exit()
        
        # cv2.imshow(self.windowname, self.mask)
        # cv2.waitKey(100)
        cv2.destroyAllWindows()
        return 
    

    def getmask(self, name='source_mask.png'):
        self._paint()
        mask_path = PATH+name
        cv2.imwrite(mask_path, self.mask)
        return mask_path



class MaskMover():
    # 将在 source 创建的 mask 移动到 target 中需要融合的位置
    # 新的 mask 应依据 target 的尺寸创建
    def __init__(self, image_path, source_mask_path):
        self.image_path = image_path
        self.source_mask_path = source_mask_path
        self.image = cv2.imread(image_path)
        self.source_mask = cv2.imread(source_mask_path)
        self.mask = np.zeros(self.image.shape) # 按照 img 尺寸新建 mask
        self.mask[np.where(self.source_mask != 0)] = 255 # 把原 mask 复制到新尺寸 mask
        self.mask_copy = self.mask.copy()

        self.status = 0 # 鼠标状态
        self.x0, self.y0 = 0, 0 # 每次操作的基准点
        self._x, self._y = 0, 0 # x, y 的总平移量

        self.windowname = "Move selected region to this pic. (s: save, r:reset, esc: exit)"


    def _blend(self, image, mask):
        res = image.copy()
        alpha = 0.7
        res[mask != 0] = res[mask != 0]*alpha + 255*(1-alpha) # 融合图像 image : mask = 0.7 : 0.3 
        # 语义分割
        res = res.astype(np.uint8)
        return res


    def _onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.status = 1
            self.x0, self.y0 = x, y
            # 检测左键按下
        elif self.status == 1 and event == cv2.EVENT_MOUSEMOVE:
            M = np.float32([[1, 0, x-self.x0],
                            [0, 1, y-self.y0]])
            self.mask = cv2.warpAffine(self.mask, M, (self.mask.shape[1], self.mask.shape[0]))
            # shape -> (h, w, c)
            # wrapAffine(img, matrix, (w, h)) 仿射变换 边界自动补偿为0（黑色）
            cv2.imshow(self.windowname, self._blend(self.image, self.mask))
            self._x += x-self.x0
            self._y += y-self.y0
            self.x0, self.y0 = x, y 
            # 实时更新移动位置（计算相对位置）
        else:
            self.status = 0
            # 检测左键释放
    

    def _move(self, name='mask_blend.png'):
        cv2.namedWindow(self.windowname)
        cv2.setMouseCallback(self.windowname, self._onMouse)

        while True:
            cv2.imshow(self.windowname, self._blend(self.image, self.mask)) # target & mask 融合可视化
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                break
            elif key == ord('r'):
                self.mask = self.mask_copy
            elif key == 27:
                cv2.destroyAllWindows()
                exit()
        
        cv2.destroyAllWindows()
        mask_path = PATH+name
        cv2.imwrite(mask_path, self._blend(self.image, self.mask))
        # 效果展示 mask_blend.png
        return 
    
    
    def getshift(self):
        return self._x, self._y


    def getmask(self, name='raw_target_mask.png'):
        self._move()
        mask_path = PATH+name
        cv2.imwrite(mask_path, self.mask)
        return mask_path



class Poisson():
    def __init__(self):
        pass


    """ OpenCV 提供的方法
    def _seamlessClone(self, obj, dst, mask, shift):
        # The location of the center of the src in the dst
        width, height, channels = dst.shape
        center = (height//2, width//2)
        # center -> (h, w) 不能越界
        # Seamlessly clone src into dst and put the results in output
        normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.NORMAL_CLONE)
        mixed_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)

        # Write results
        cv2.imwrite(PATH+"opencv-normal-clone-example.jpg", normal_clone)
        cv2.imwrite(PATH+"opencv-mixed-clone-example.jpg", mixed_clone)
    """

    def normalSeamlessClone(self, source, target, mask, shift):
        y_range, x_range = target.shape[:-1] # height, width

        M = np.float32([[1, 0, shift[0]],
                        [0, 1, shift[1]]])
        source = cv2.warpAffine(source, M, (x_range, y_range)) # dsize(width, height)
        # 把 source 按照 mask 移动到合适位置，用 mask 匹配融合区域

        # laplacian 矩阵
        L = laplacian_matrix(y_range, x_range)
        mat_A = L.copy()
        # 修改融合区域外的矩阵信息
        for y in range(1, y_range-1):
            for x in range(1, x_range-1):
                if mask[y, x] == 0:
                    k = x + y * x_range
                    mat_A[k, k] = 1
                    mat_A[k, k-1] = mat_A[k, k+1] = 0
                    mat_A[k, k-x_range] = mat_A[k, k+x_range] = 0
                    # 4 -> 1, others -> 0

        res = np.zeros(target.shape)
        mask_flat = mask.flatten()
        for channel in range(target.shape[2]):
            source_flat = source[:,:,channel].flatten()
            target_flat = target[:,:,channel].flatten()
            mat_B = L.dot(source_flat) # 生成 nabla g
            mat_B[mask_flat == 0] = target_flat[mask_flat == 0] # 保留 mask 外的图像
            # 解整张图
            # mat_A = mat_A.tocsc()
            tmp = scipy.sparse.linalg.spsolve(mat_A, mat_B)
            tmp = tmp.reshape((y_range, x_range))
            tmp[tmp > 255] = 255
            tmp[tmp < 0] = 0
            res[:,:,channel] = tmp.astype('uint8')

        return res


    def mixedSeamlessClone(self, source, target, mask, shift):
        y_range, x_range = target.shape[:-1] # height, width

        M = np.float32([[1, 0, shift[0]],
                        [0, 1, shift[1]]])
        source = cv2.warpAffine(source, M, (x_range, y_range))

        s_grad_x, s_grad_y = get_gradient(source)
        t_grad_x, t_grad_y = get_gradient(target)

        for c in range(target.shape[2]):
            for y in range(y_range):
                for x in range(x_range):
                    if mask[y, x] != 0:
                        if abs(t_grad_x[y, x, c]) + abs(t_grad_y[y, x, c]) < abs(s_grad_x[y, x, c]) + abs(s_grad_y[y, x, c]):
                            t_grad_x[y, x, c] = s_grad_x[y, x, c]
                            t_grad_y[y, x, c] = s_grad_y[y, x, c]
                            # mixed gradient 按梯度绝对值分配 gradient
        
        lap = get_laplacian(t_grad_x, t_grad_y)
        # laplacian expression of result image

        mat_A = laplacian_matrix(y_range, x_range)

        # 修改边界点
        lap[mask == 0] = target[mask == 0]
        for y in range(1, y_range-1):
            for x in range(1, x_range-1):
                if mask[y, x] == 0:
                    k = x + y * x_range
                    mat_A[k, k] = 1
                    mat_A[k, k-1] = mat_A[k, k+1] = 0
                    mat_A[k, k-x_range] = mat_A[k, k+x_range] = 0

        res = np.zeros(source.shape)
        mask_flat = mask.flatten()
        for channel in range(target.shape[2]):
            mat_B = lap[:,:,channel].flatten()
            
            tmp = scipy.sparse.linalg.spsolve(mat_A, mat_B)
            tmp = tmp.reshape((y_range, x_range))
            tmp[tmp > 255] = 255
            tmp[tmp < 0] = 0
            res[:,:,channel] = tmp.astype('uint8')
        
        return res


    def textureFlatten(self, source, mask, alpha = 4):
        y_range, x_range = source.shape[:-1]

        s_grad_x, s_grad_y = get_gradient(source)

        for c in range(source.shape[2]):
            for y in range(y_range):
                for x in range(x_range):
                    if mask[y, x] != 0:
                        if abs(s_grad_x[y, x, c]) < alpha:
                            s_grad_x[y, x, c] = 0
                        if abs(s_grad_y[y, x, c]) < alpha:
                            s_grad_y[y, x, c] = 0
                            # 忽略 < alpha 的梯度
        
        lap = get_laplacian(s_grad_x, s_grad_y)

        mat_A = laplacian_matrix(y_range, x_range)

        # 修改边界点
        lap[mask == 0] = source[mask == 0]
        for y in range(1, y_range-1):
            for x in range(1, x_range-1):
                if mask[y, x] == 0:
                    k = x + y * x_range
                    mat_A[k, k] = 1
                    mat_A[k, k-1] = mat_A[k, k+1] = 0
                    mat_A[k, k-x_range] = mat_A[k, k+x_range] = 0

        res = np.zeros(source.shape)
        mask_flat = mask.flatten()
        for channel in range(source.shape[2]):
            mat_B = lap[:,:,channel].flatten()
            
            tmp = scipy.sparse.linalg.spsolve(mat_A, mat_B)
            tmp = tmp.reshape((y_range, x_range))
            tmp[tmp > 255] = 255
            tmp[tmp < 0] = 0
            res[:,:,channel] = tmp.astype('uint8')
        
        return res



if __name__ == "__main__":
    print("Launch: 'Poisson image editing'.")
    # PATH = input("Input the folder PATH of main.py : ")
    # 读取图像
    source_path = PATH + 'source.jpg'
    target_path = PATH + 'target.jpg'
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    
    # 创建 mask
    source_mask = MaskPainter(source_path)
    source_mask_path = source_mask.getmask()
    print("Saved source mask: " + source_mask_path)
    # source_mask_path = PATH+'source_mask.png'

    target_mask = MaskMover(target_path, source_mask_path)
    target_mask_path = target_mask.getmask()
    print("Saved target mask: " + target_mask_path)

    shift = target_mask.getshift()
    print("shift info: ", shift)

    mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
    result = Poisson().mixedSeamlessClone(source, target, mask, shift)
    # result = Poisson().textureFlatten(source, mask)
    cv2.imwrite(PATH+'result.png', result)
    # 输出图片