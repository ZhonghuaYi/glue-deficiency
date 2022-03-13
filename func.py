import numpy as np
import cv2 as cv
from math import sqrt, pow


def get_histogram(in_pic, scale=256):
    """
    获取正则化的直方图
    :param in_pic: 输入图像
    :param scale: 输入图像的灰阶
    :return: 直方图(numpy)
    """
    histogram = np.zeros(scale)
    pic_size = in_pic.size
    for i in in_pic.flat:
        histogram[i] += 1

    histogram /= pic_size
    return histogram


def cdf(in_pic_histogram):
    """
    获取直方图的累积分布函数(CDF)
    :param in_pic_histogram: 直方图
    :return: 累积分布函数(numpy)
    """
    scale = in_pic_histogram.size
    transform = np.zeros(scale)
    temp = 0.
    for i in range(scale):
        temp += in_pic_histogram[i]
        transform[i] = temp

    return (transform * (scale - 1)).astype(np.uint8)


def match_histogram(in_pic, match):
    """
    直方图匹配
    :param in_pic: 输入图像
    :param match: 被匹配的直方图
    :return: 输出图像(numpy)
    """
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
        ind = 0
        for n in range(256):
            if min_val > diff_cdf[m][n]:
                min_val = diff_cdf[m][n]
                ind = n
        lut[m] = ind

    result = cv.LUT(in_pic, lut)
    return result


def refer_generate(dir_path, flag=0):
    """
    将文件夹下的参考图片生成一个生成器
    :return: 参考样本的生成器
    """
    import os
    files = os.listdir(dir_path)
    for file in files:
        if file[:5] != "refer":
            continue
        file_path = dir_path + "/" + file
        img = cv.imread(file_path, flag)
        yield img


def sample_generate(dir_path, sample_list=None, flag=0):
    """
    将文件夹下的样本图片生成一个生成器
    :return: 样本的生成器
    """
    import os
    files = os.listdir(dir_path)
    if sample_list:
        for s in sample_list:
            img_path = dir_path + s
            img = cv.imread(img_path, 0)
            yield img
    else:
        for file in files:
            if file[:6] != "sample":
                continue
            img_path = dir_path + "/" + file
            img = cv.imread(img_path, flag)
            yield img


def threshold_segment(img, threshold):
    """
    阈值分割，将阈值范围内的灰度置零，其他灰度置255
    :param img: 输入图像
    :param threshold: 阈值
    :return: 输出图像
    """
    out = np.where(img < threshold, 0, 255)
    # out = img.copy()
    # for i in np.nditer(out, op_flags=['readwrite']):
    #     if threshold[0] <= i < threshold[1]:
    #         i[...] = 0
    #     else:
    #         i[...] = 255

    return out.astype(img.dtype)


def area_segment(img, pre_area_num):
    """
    图像的区域分割
    :param pre_area_num: 初始假设的区域数量，应略多于实际区域数量
    :param img: 输入图像
    :return: （区域数值，区域面积）(ndarray)， 区域起始坐标
    """

    def neighbor_expand(x, y, value, region_area):
        """
        此函数用于递归确定区域
        :param region_area: 当前区域累计面积
        :param x: x坐标
        :param y: y坐标
        :param value: 区域的新值，用于和原来的值区分开
        :return: 累计面积
        """
        img[x][y] = value
        region_area += 1
        # ind为遍历的邻域顺序
        ind = ((x, y+ 1), (x + 1, y), (x, y - 1), (x - 1, y))
        for x1, y1 in ind:
            if (x1 < 0) or (x1 >= img_shape[0]) or (y1 < 0) or (y1 >= img_shape[1]):
                continue
            if img[x1][y1] == 0:
                region_area = neighbor_expand(x1, y1, value, region_area)
        return region_area

    region_value = np.linspace(start=2, stop=255, num=pre_area_num, dtype=np.uint8)
    region_num = 0
    regions = []
    region_start = []
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i][j] == 0:
                region = [0, 0]  # 区域的数值与面积
                region_start.append((i, j))
                region[1] = neighbor_expand(i, j, region_value[region_num], region[1])
                region[0] = region_value[region_num]
                region_num += 1
                regions.append(region)

    return np.array(regions), region_start


def template_generate(refer_sample, x=(), y=(), flag="canny", canny=(50, 120), thresh=50):
    """
    从参考样本中截取某个区域作为模板
    :param refer_sample: 参考样本集
    :param x: 区域的row方向区间
    :param y: 区域的col方向区间
    :param flag: 获取模板的方式，canny或者thresh
    :param canny: canny的两个阈值
    :param thresh: thresh的阈值
    :return: 模板图像
    """
    # 求所有图像的平均
    refer_count = 1
    t = np.empty((1, 1))
    for image in refer_sample:
        # scale = min(image.shape) / 500
        # new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
        # image = cv.resize(image, new_size)
        image = image[x[0]:x[1], y[0]:y[1]]
        image = image.astype(np.float32)
        if refer_count == 1:
            t = image
        t += image
        refer_count += 1

    t /= refer_count
    t = t.astype(np.uint8)

    if flag == "canny":
        # 对图像进行高斯平滑
        t = cv.GaussianBlur(t, (7, 7), sigmaX=1)
        # Canny法提取图像边缘
        t = cv.Canny(t, canny[0], canny[1])

    elif flag == "thresh":
        t = cv.medianBlur(t, 3)
        __, t = cv.threshold(t, thresh, 255, cv.THRESH_BINARY)

    return t


def gaussian_pyramid(img, flag, num):
    """
    返回图像金字塔
    :param img: 原图像
    :param flag: "up"或者"down"，表示上采样或者下采样
    :param num: 金字塔层数
    :return: list列表，图像大小由小到大或者由大到小，取决于采样方向
    """
    image = img.copy()
    pyramid = [image]
    if flag == "up":
        for i in range(num - 1):
            image = cv.pyrUp(image)
            pyramid.append(image)
    elif flag == "down":
        for i in range(num - 1):
            image = cv.pyrDown(image)
            pyramid.append(image)

    return pyramid[::-1]


def nearest_point(point, pt_set):
    min_distance = 1000000
    index = 0
    for i in range(len(pt_set)):
        distance = sqrt(pow(point.pt[0]-pt_set[i].pt[0], 2) + pow(point.pt[1]-pt_set[i].pt[1], 2))
        if distance < min_distance:
            min_distance = distance
            index = i

    return min_distance, index


def key_point_match(kp_t, kp_img, th, des_t=np.empty((0, 0)), des_img=np.empty((0, 0))):
    matchs = []
    if len(des_t) and len(des_img):
        new_kp_t = []
        new_kp_img = []
        new_des_t = []
        new_des_img = []
        for i in range(len(kp_t)):
            distance_1, index_1 = nearest_point(kp_t[i], kp_img)
            distance_2, index_2 = nearest_point(kp_img[index_1], kp_t)
            if index_2 == i and distance_1 <= th:
                match = cv.DMatch()
                match.distance = distance_1
                match.queryIdx = i
                match.trainIdx = index_1
                match.imgIdx = 0
                matchs.append(match)
                new_kp_t.append(kp_t[i])
                new_kp_img.append(kp_img[index_1])
                new_des_t.append(des_t[i])
                new_des_img.append(des_img[index_1])

        new_des_t = np.array(new_des_t)
        new_des_img = np.array(new_des_img)
        return new_kp_t, new_kp_img, new_des_t, new_des_img, matchs
    else:
        for i in range(len(kp_t)):
            distance_1, index_1 = nearest_point(kp_t[i], kp_img)
            distance_2, index_2 = nearest_point(kp_img[index_1], kp_t)
            if index_2 == i and distance_1 <= th:
                match = cv.DMatch()
                match.distance = distance_1
                match.queryIdx = i
                match.trainIdx = index_1
                match.imgIdx = 0
                matchs.append(match)

        return matchs


def draw_line(drawing, lines):
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(drawing, (x1, y1), (x2, y2), 255)

    return drawing


def defect1_hough_line(img):
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 5, minLineLength=5, maxLineGap=2)
    out = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 != x2:
            if y2 >= y1:
                k = (y2 - y1) / (x2 - x1)
            else:
                k = (y1 - y2) / (x1 - x2)

            if 0.3 < k < 3:
                out.append(line[0].tolist())

    return out


def defect2_hough_line(img):
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 5, minLineLength=3, maxLineGap=5)
    out = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        divide = x2 - x1
        if divide != 0:
            k = abs((y2-y1) / divide)
            if k < 0.5:
                out.append(line[0].tolist())

    return out


def result_explain(result, n):
    message = f"区域{n}："
    if result == 0:
        message = message + f"不正常"
    elif result == 1:
        message = message + f"正常"
    elif result == 2:
        message = message + f"没有目标区域"
    else:
        message = message + f"错误的结果码"

    print(message)


