import cv2 as cv
import numpy as np

import func

test0 = cv.imread('test000.bmp', 0)
test0 = cv.GaussianBlur(test0, (5, 5), 5)
test0 = cv.equalizeHist(test0)
#test0 = cv.Canny(test0, 40, 110)


cv.imshow('img', test0)
cv.waitKey(0)
