import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import Spatial


if __name__ == '__main__':

    refer_img_list = ['test000.bmp']
    test_img_list = ['test001.bmp', 'test002.bmp']
    target_hist = np.loadtxt('target_hist.csv', delimiter=' ')

    img = cv.imread(test_img_list[0], 0)
    # img = cv.medianBlur(img, 5)
    img = cv.GaussianBlur(img, (5, 5), 1.7)
    # img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0)
    for i in np.nditer(img, op_flags=['readwrite']):
        if i < 38:
            i[...] = 0
        else:
            i[...] = 255
    # img = Spatial.histogram_matching(img, target_hist)
    # img = cv.pyrDown(img)
    # img = cv.Canny(img, 70, 200)
    # hist = np.bincount(img.ravel(), minlength=256)
    # norm_hist = cv.normalize(hist.astype(np.float32), None, 1, 0, cv.NORM_INF)
    # plt.plot(norm_hist)
    # plt.show()
    cv.imshow('img', img)
    cv.waitKey(0)

# for test in (test0, test1):
#     for i in np.nditer(test, op_flags=['readwrite']):
#         if i < 80:
#             i[...] = 80
#         else:
#             i[...] = 255

# test0 = cv.Canny(test0, 60, 130)

