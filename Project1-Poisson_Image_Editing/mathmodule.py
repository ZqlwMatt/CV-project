# Math Module for posiison image editing

import cv2
import numpy as np
import scipy.sparse

def get_gradient(img):
    kernel_x = np.array([[0, 0, 0],
                         [0,-1, 1],
                         [0, 0, 0]])
    kernel_y = np.array([[0, 0, 0],
                         [0,-1, 0],
                         [0, 1, 0]])
    grad_x = cv2.filter2D(img, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_32F, kernel_y)
    return grad_x, grad_y


def get_laplacian(grad_x, grad_y):
    kernel_x = np.array([[0, 0, 0],
                         [-1,1, 0],
                         [0, 0, 0]])
    kernel_y = np.array([[0,-1, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
    lap_x = cv2.filter2D(grad_x, cv2.CV_32F, kernel_x)
    lap_y = cv2.filter2D(grad_y, cv2.CV_32F, kernel_y)
    return lap_x + lap_y


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-4)
    mat_D.setdiag(1, -1)
    mat_D.setdiag(1, 1)
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    mat_A.setdiag(1, -m)
    mat_A.setdiag(1, m)
    
    return mat_A.tocsc()


