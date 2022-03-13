"""
    这个文件里的是有关图像的分割方法
"""

import cv2 as cv
import numpy as np

import func


def template_match(image, target_template, canny=(50, 120), flag=0):
    """
    模板匹配
    :param image: 原图像
    :param target_template: 目标区域的模板
    :param canny: Canny法的低阈值和高阈值
    :return: 目标区域与模板的相关系数
    """
    template_shape = target_template.shape
    # 第一步，将图像缩放到一个统一的大小（较小边为500像素）
    scale = min(image.shape) / 500
    new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
    image = cv.resize(image, new_size)

    # 第二步，对图像进行高斯平滑
    image = cv.GaussianBlur(image, (3, 3), sigmaX=1)

    # 第三步，Canny法提取图像边缘
    image = cv.Canny(image, canny[0], canny[1])

    # 获取模板和图像的三层下取样金字塔
    t_pyramid = func.gaussian_pyramid(target_template, "down", 3)
    img_pyramid = func.gaussian_pyramid(image, "down", 3)

    # 通过从金字塔的上层开始进行模板匹配，找到最匹配的角度，误差在0.2度
    angle_interval = (-45, 45)  # 模板旋转的角度范围
    angle_num = (10, 11, 11)  # 对每层模板旋转的角度个数，元组长度应与t_pyramid相同
    match_angle = 0  # 最匹配的角度
    for i in range(len(t_pyramid)):
        max_ccoeff = 0
        angle_step = (angle_interval[1] - angle_interval[0]) / (angle_num[i] - 1)
        for j in range(angle_num[i]):
            # 得到角度
            angle = angle_interval[0] + j * angle_step
            # 旋转模板
            M = cv.getRotationMatrix2D((t_pyramid[i].shape[1] / 2, t_pyramid[i].shape[0] / 2), angle, 1)
            t = cv.warpAffine(t_pyramid[i], M, (t_pyramid[i].shape[1], t_pyramid[i].shape[0]))
            # 匹配区域
            res = cv.matchTemplate(img_pyramid[i], t, cv.TM_CCOEFF_NORMED)  # 使用的方法是相关系数
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            CCOEFF = max_val  # 记录此时最匹配区域的相关系数
            if CCOEFF >= max_ccoeff:
                max_ccoeff = CCOEFF
                match_angle = angle
        if i < len(t_pyramid) - 1:
            angle_interval = (match_angle - angle_step, match_angle + angle_step)

    # print(match_angle)
    # 将原图像旋转-match_angle度并与原模板匹配
    M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), -match_angle, 1)
    image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    compare = np.ones(image.shape, dtype=image.dtype) * 128
    image = np.array(image >= compare).astype(image.dtype) * 255
    res = cv.matchTemplate(image, target_template, cv.TM_CCOEFF_NORMED)  # 使用的方法是相关系数
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    CCOEFF = max_val  # 记录此时最匹配区域的相关系数
    left_top = max_loc  # 最匹配模板的区域的左上角坐标，为宽和高，不是x和y坐标
    image = image[left_top[1]:left_top[1] + template_shape[0],
                  left_top[0]:left_top[0] + template_shape[1]]

    return CCOEFF, image


def hausdorff_match(template, image, canny=(50, 120)):
    from hausdorff import hausdorff_distance
    # 第一步，将图像缩放到一个统一的大小（较小边为500像素）
    scale = min(image.shape) / 500
    new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
    image = cv.resize(image, new_size)

    # 第二步，对图像进行高斯平滑
    image = cv.GaussianBlur(image, (3, 3), sigmaX=1)

    # 第三步，Canny法提取图像边缘
    image = cv.Canny(image, canny[0], canny[1])
    t_shape = template.shape
    img_shape = image.shape
    min_distance = np.inf
    min_ind = [0, 0]
    for i in range(0, img_shape[0] - t_shape[0]):
        for j in range(0, img_shape[1] - img_shape[1]):
            h_distance = hausdorff_distance(template, image[i:i + t_shape[0], j:j + t_shape[1]])
            if h_distance < min_distance:
                min_distance = h_distance
                min_ind = [i, j]

    i, j = min_ind
    return image[i:i + t_shape[0], j:j + t_shape[1]]


def threshold_segment(image, area_percent, pre_area_num, structure_element):
    """
    阈值分割
    :param image: 原图像
    :param area_percent: 灰度阈值以下的灰度所占图像的面积比
    :param pre_area_num: 预设的最大区域数量
    :param structure_element: 用于形态学计算的结构元素
    :return:
    """
    # 第一步
    # 在参考图像中，当手动阈值在37时，阈值分割效果明显。于是考虑到灰度小于37的区域大概面积占比是0.3，
    # 于是将图像中灰度值较低的30%区域分割出来。这里利用了cdf，它本身是直方图的累积分布，因此只需要寻找
    # cdf中最接近0.3的位置，其索引即是能够将30%灰度比较低的区域分割出来的阈值
    image = image[0:800, 0:800]
    hist = func.get_histogram(image)
    img_cdf = func.cdf(hist) / 255.  # 正则化后的cdf，映射到了（0，1）范围
    index = (np.abs(img_cdf - area_percent)).argmin()  # index即为阈值
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
    regions, region_start = func.area_segment(image, pre_area_num)

    # 第四步，将面积第二的区域提取出来
    ind = np.argsort(regions[:, 1])[-2]
    target_region_value = regions[ind, 0]  # 获取到面积第二的区域的值
    region_area = regions[ind, 1]  # 将该区域的面积记录下来
    compare = np.ones(image.shape, dtype=image.dtype) * target_region_value
    image = np.array(image != compare).astype(image.dtype) * 255  # 将数值为target_area_value的区域分离出来

    return region_area, image
