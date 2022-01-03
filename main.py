import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from func import *


def class1_defect1():
    image_root = './image/class1_defect1/'
    refer_images = ['refer000.BMP']  # 参考图像
    defect_images = ['defect000.BMP', 'defect001.BMP']  # 缺陷图像
    pre_area_num = 12

    refer_list = [image_root + image for image in refer_images]
    defect_list = [image_root + image for image in defect_images]
    sample = sample_generate(refer_list, defect_list)  # 样本的生成器
    count = 1  # 图像的计数
    structure_element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # 用于形态学计算的矩形结构元素
    target_region_areas = []
    start_time = time.time()
    for image in sample:
        # 第一步
        # 在参考图像中，当手动阈值在37时，阈值分割效果明显。于是考虑到灰度小于37的区域大概面积占比是0.3，
        # 于是将图像中灰度值较低的30%区域分割出来。这里利用了cdf，它本身是直方图的累积分布，因此只需要寻找
        # cdf中最接近0.3的位置，其索引即是能够将30%灰度比较低的区域分割出来的阈值
        image = image[0:800, 0:800]
        hist = get_histogram(image)
        img_cdf = cdf(hist) / 255.  # 正则化后的cdf，映射到了（0，1）范围
        index = (np.abs(img_cdf - 0.3)).argmin()  # index即为阈值
        # print("阈值：" + str(index))

        # 第二步
        # 得到阈值后，对图像进行阈值分割，然后对不规则区域应用中值滤波平滑。
        # 每次平滑后，需要通过膨胀背景从而腐蚀物体，以使得目标区域能够更容易被分离
        image = cv.medianBlur(image, 9)
        image = cv.dilate(image, structure_element)
        # resize到400x400后再中值滤波，然后腐蚀
        scale = min(image.shape) / 400
        new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
        image = cv.resize(image, new_size)
        image = cv.medianBlur(image, 3)
        image = cv.dilate(image, structure_element)
        # resize到70x70后，阈值分割（这里只能最大缩小到70，否则会爆栈，这和内存有关）
        scale = min(image.shape) / 70
        new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
        image = cv.resize(image, new_size)
        # image = threshold_segment(image, index)  # 自己写的阈值分割函数
        th, image = cv.threshold(image, index, 255, cv.THRESH_BINARY)  # 官方的阈值分割函数

        # 第三步，将图像分成若干个区域
        regions, region_start = area_segment(image, pre_area_num)
        print(regions)  # 输出所有检测到的区域的数值和面积

        # 第四步，将面积第二的区域提取出来
        ind = np.argsort(regions[:, 1])[-2]
        target_region_value = regions[ind, 0]  # 获取到面积第二的区域的值
        target_region_areas.append(regions[ind, 1])  # 将该区域的面积记录下来
        compare = np.ones(image.shape, dtype=image.dtype) * target_region_value
        image = np.array(image != compare).astype(image.dtype) * 255  # 将数值为target_area_value的区域分离出来

        # 显示最终分离出的区域的图像
        window_name = 'img' + str(count)  # 图像窗口的名称
        cv.imshow(window_name, image)
        count += 1  # 图像计数加一

    print(target_region_areas)  # 输出区域的面积
    end_time = time.time()
    print('运行时间：', end_time - start_time)
    cv.waitKey(0)


def class1_defect2():
    # 读取样本
    image_root = './image/class1_defect2/'
    refer_images = ['refer000.BMP']
    defect_images = ['defect000.BMP']
    refer_list = [image_root + image for image in refer_images]
    defect_list = [image_root + image for image in defect_images]
    sample = sample_generate(refer_list, defect_list)  # 样本的生成器

    # 读取模板图像
    template_path = image_root + 'target_template.BMP'
    target_template = cv.imread(template_path, 0)
    template_shape = target_template.shape

    # 图像的计数
    count = 1

    # 记录图像与模板最匹配的区域的相关系数
    ccoeff = []

    start_time = time.time()
    for image in sample:
        # 第一步，将图像缩放到一个统一的大小（较小边为500像素）
        scale = min(image.shape) / 500
        new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
        image = cv.resize(image, new_size)

        # 第二步，对图像进行高斯平滑
        image = cv.GaussianBlur(image, (3, 3), sigmaX=1)

        # 第三步，Canny法提取图像边缘
        image = cv.Canny(image, 50, 120)

        # 第四步，通过模板匹配，找到目标区域
        res = cv.matchTemplate(image, target_template, cv.TM_CCOEFF_NORMED)  # 使用的方法是相关系数
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        ccoeff.append(max_val)  # 记录此时最匹配区域的相关系数
        left_top = max_loc  # 最匹配模板的区域的左上角坐标，为宽和高，不是x和y坐标
        image = image[left_top[1]:left_top[1] + template_shape[0],
                      left_top[0]:left_top[0] + template_shape[1]]

        # 显示目标区域图像
        window_name = 'img' + str(count)  # 图像窗口的名称
        cv.imshow(window_name, image)
        count += 1  # 图像计数加一

    end_time = time.time()
    print('相关系数：', ccoeff)
    print('运行时间：', end_time - start_time)
    cv.waitKey(0)


if __name__ == '__main__':
    class1_defect1()
