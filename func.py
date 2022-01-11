import numpy as np
import cv2 as cv


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


def sample_generate(refer_list, defect_list):
    """
    产生一个所有样本的生成器
    :return: 所有样本的生成器
    """
    sample_list = refer_list + defect_list
    for sample_path in sample_list:
        img = cv.imread(sample_path, 0)
        yield img


def refer_sample_generate(refer_list):
    """
    产生一个参考样本的生成器
    :return: 参考样本的生成器
    """
    for i in range(len(refer_list)):
        img = cv.imread(refer_list[i], 0)
        yield img


def defect_sample_generate(defect_list):
    """
    产生一个缺陷样本的生成器
    :return: 缺陷样本的生成器
    """
    for i in range(len(defect_list)):
        img = cv.imread(defect_list[i], 0)
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


def template_generate(refer_sample, template_size, x=(), y=(), canny=(50, 120)):
    # 第一步，对每张图像缩放到500，然后求所有图像的平均
    refer_count = 1
    t = refer_sample.__next__()
    scale = min(t.shape) / template_size
    new_size = round(t.shape[1] / scale), round(t.shape[0] / scale)  # 这里的size指宽度和高度
    t = cv.resize(t, new_size)
    t = t[x[0]:x[1], y[0]:y[1]]
    t = t.astype(np.float32)
    for image in refer_sample:
        scale = min(image.shape) / template_size
        new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
        image = cv.resize(image, new_size)
        image = image[x[0]:x[1], y[0]:y[1]]
        image = image.astype(np.float32)
        t += image
        refer_count += 1

    t /= refer_count
    t = t.astype(np.uint8)

    # 第二步，对图像进行高斯平滑
    t = cv.GaussianBlur(t, (3, 3), sigmaX=1)

    # 第三步，Canny法提取图像边缘
    t = cv.Canny(t, canny[0], canny[1])
    return t


def draw_line(drawing, lines):
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(drawing, (x1, y1), (x2, y2), 255)

    return drawing


def defect1_hough_line(img):
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 5, minLineLength=5, maxLineGap=2)
    out = []
    for line in lines:
        k = -1
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
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 5, minLineLength=7, maxLineGap=2)
    out = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        divide = x2 - x1
        if divide != 0:
            k = abs((y2-y1) / divide)
            if k < 0.2:
                out.append(line[0].tolist())

    return out
