import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import func

test0 = cv.imread('test000.bmp', 0)
# test0 = cv.resize(test0, (256, 256))
test0 = cv.medianBlur(test0, 7)
test0 = cv.equalizeHist(test0)

test1 = cv.imread('test001.bmp', 0)
# test1 = cv.resize(test1, (256, 256))
test1 = cv.medianBlur(test1, 7)
test1 = cv.equalizeHist(test1)
# test0 = cv.GaussianBlur(test0, (3, 3), 2)
# test = cv.adaptiveThreshold(test0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0)
# plt.hist(test0.ravel(), 256, [0, 256])
# plt.show()
for test in (test0, test1):
    for i in np.nditer(test, op_flags=['readwrite']):
        if i< 80:
            i[...] = 80
        else:
            i[...] = 255

# test0 = cv.Canny(test0, 60, 130)


cv.imshow('img', test0)
cv.imshow('img1', test1)
cv.waitKey(0)
