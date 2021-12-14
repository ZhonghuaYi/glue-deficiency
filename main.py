import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import Spatial

REFER_LIST = ['test000.bmp']
DEFECT_LIST = ['test001.bmp', 'test002.bmp']


def get_histogram(in_pic, scale=256):
    histogram = np.zeros(scale)
    pic_size = in_pic.size
    for i in in_pic.flat:
        histogram[i] += 1

    histogram /= pic_size
    return histogram


# get the cumulative distribution function(CDF) of in_pic.
def cdf(in_pic_histogram):
    scale = in_pic_histogram.size
    transform = np.zeros(scale)
    temp = 0.
    for i in range(scale):
        temp += in_pic_histogram[i]
        transform[i] = temp

    return (transform * (scale - 1)).astype(np.uint8)


def match_histogram(in_pic, match):
    in_histogram = get_histogram(in_pic)
    in_cdf = cdf(in_histogram)
    match_cdf = cdf(match)
    # 构建累积概率误差矩阵
    diff_cdf = np.zeros((256, 256))
    for k in range(256):
        for j in range(256):
            diff_cdf[k][j] = np.abs(int(in_cdf[k]) - int(match_cdf[j]))

    # 生成映射表
    lut = np.zeros((256, ), dtype=np.uint8)
    for m in range(256):
        min_val = diff_cdf[m][0]
        index = 0
        for n in range(256):
            if min_val > diff_cdf[m][n]:
                min_val = diff_cdf[m][n]
                index = n
        lut[m] = index
    
    result = cv.LUT(in_pic, lut)
    return result


def refer_sample_generate():
    for i in range(len(REFER_LIST)):
        img = cv.imread(REFER_LIST[i], 0)
        yield img


def defect_sample_generate():
    for i in range(len(DEFECT_LIST)):
        img = cv.imread(DEFECT_LIST[i], 0)
        yield img


def threshold_segment(img, threshold):
    out = img.copy()
    out = cv.GaussianBlur(out, (5, 5), 1.7)
    for i in np.nditer(out, op_flags=['readwrite']):
        if i < threshold:
            i[...] = 0
        else:
            i[...] = 255

    return out


if __name__ == '__main__':
    refer_sample = refer_sample_generate()
    image = refer_sample.__next__()
    image = threshold_segment(image, 37)
    cv.imshow('img', image)
    cv.waitKey(0)

