import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import Spatial

REFER_LIST = ['test000.bmp']  # 参考图像
DEFECT_LIST = ['test001.bmp', 'test002.bmp']  # 缺陷图像
PRE_AREA_NUM = 12


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


def sample_generate():
    """
    产生一个所有样本的生成器
    :return: 所有样本的生成器
    """
    sample_list = REFER_LIST + DEFECT_LIST
    for sample_path in sample_list:
        img = cv.imread(sample_path, 0)
        yield img


def refer_sample_generate():
    """
    产生一个参考样本的生成器
    :return: 参考样本的生成器
    """
    for i in range(len(REFER_LIST)):
        img = cv.imread(REFER_LIST[i], 0)
        yield img


def defect_sample_generate():
    """
    产生一个缺陷样本的生成器
    :return: 缺陷样本的生成器
    """
    for i in range(len(DEFECT_LIST)):
        img = cv.imread(DEFECT_LIST[i], 0)
        yield img


def threshold_segment(img, threshold):
    """
    阈值分割，将阈值范围内的灰度置零，其他灰度置255
    :param img: 输入图像
    :param threshold: 阈值范围（元组）
    :return: 输出图像
    """
    out = img.copy()
    # out = cv.GaussianBlur(out, (5, 5), 1.7)
    for i in np.nditer(out, op_flags=['readwrite']):
        if threshold[0] <= i < threshold[1]:
            i[...] = 0
        else:
            i[...] = 255

    return out


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
        ind = ((x, y+1), (x+1, y), (x, y-1), (x-1, y))
        for x1, y1 in ind:
            if (x1 < 0) or (x1 >= img_shape[0]) or (y1 < 0) or (y1 >= img_shape[1]):
                continue
            if img[x1][y1] == 0:
                region_area = neighbor_expand(x1, y1, value, region_area)
        return region_area

    area_value = np.linspace(start=2, stop=255, num=pre_area_num, dtype=np.uint8)
    area_num = 0
    regions = []
    area_start = []
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if img[i][j] == 0:
                area = [0, 0]  # 区域的数值与面积
                area_start.append((i, j))
                area[1] = neighbor_expand(i, j, area_value[area_num], area[1])
                area[0] = area_value[area_num]
                area_num += 1
                regions.append(area)

    return np.array(regions), area_start


if __name__ == '__main__':
    sample = sample_generate()  # 样本的生成器
    count = 1  # 图像的计数
    structure_element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # 用于形态学计算的矩形结构元素
    target_region_areas = []
    for image in sample:
        '''
        在参考图像中，当手动阈值在37时，阈值分割效果明显。于是考虑到灰度小于37的区域大概面积占比是0.3，
        于是将图像中灰度值较低的30%区域分割出来。这里利用了cdf，它本身是直方图的累积分布，因此只需要寻找
        cdf中最接近0.3的位置，其索引即是能够将30%灰度比较低的区域分割出来的阈值
        '''
        hist = get_histogram(image)
        img_cdf = cdf(hist) / 255.
        index = (np.abs(img_cdf - 0.3)).argmin()  # index即为阈值
        # print("阈值：" + str(index))

        '''
        得到阈值后，对图像进行阈值分割，然后对不规则区域应用中值滤波平滑。
        每次平滑后，需要通过膨胀背景从而腐蚀物体，以使得目标区域能够更容易被分离
        '''
        image = threshold_segment(image, (0, index))

        image = cv.medianBlur(image, 9)
        image = cv.dilate(image, structure_element)

        image = cv.resize(image, (400, 400))
        image = cv.medianBlur(image, 3)
        image = cv.dilate(image, structure_element)

        image = cv.resize(image, (70, 70))
        compare = np.ones(image.shape, dtype=image.dtype) * 255
        image = np.array(image == compare).astype(image.dtype) * 255
        del compare

        '''
        将图像分成若干个区域
        '''
        areas, areas_start = area_segment(image, PRE_AREA_NUM)

        '''
        当区域面积为1时，该区域为无效区域，将该区域置为背景，即清除该区域
        '''
        invalid_area_start = np.where(areas[:, 1] == 1)[0].tolist()
        for i in invalid_area_start:
            ind = areas_start[i]
            image[ind[0]][ind[1]] = 255

        '''
        将面积第二的区域提取出来
        '''
        ind = np.argsort(areas[:, 1])[-2]
        target_area_value = areas[ind, 0]  # 获取到面积第二的区域的值
        target_region_areas.append(areas[ind, 1])  # 将该区域的面积记录下来
        compare = np.ones(image.shape, dtype=image.dtype) * target_area_value
        image = np.array(image != compare).astype(image.dtype) * 255  # 将数值为target_area_value的区域分离出来

        # print(areas)
        # print(areas_start)
        # print(invalid_area_start)

        window_name = 'img' + str(count)
        cv.imshow(window_name, image)
        count += 1

    print(target_region_areas)  # 输出区域的面积
    cv.waitKey(0)

