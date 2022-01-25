"""
    这个文件的函数利用图像的特征判断图像是否有缺陷
"""


def correlation(ccoeff):
    """
    用相关系数判断图像是否有缺陷
    :param ccoeff: 相关系数
    :return: True表示图像没有缺陷
    """
    if ccoeff < 0.6:
        return False

    else:
        return True


def region_area(area, normal_area):
    """
    用检测到的区域与没有缺陷时的区域面积比值判断图像是否有缺陷
    :param area: 检测到的区域面积
    :param normal_area: 没有缺陷时的区域的面积
    :return: True表示图像没有缺陷
    """
    if float(area) / normal_area < 0.9:
        return False

    else:
        return True


def defect1_hough(img_shape, lines, max_lines):
    """
    使用hough直线检测得到的直线判断图像是否有缺陷
    :param img_shape: 图像shape
    :param lines: 所有直线
    :param max_lines: 关注的区域内允许的最大直线数
    :return: True表示图像没有缺陷
    """
    line_count = 0
    feature_zone = img_shape[1] / 3, img_shape[1] * 2 / 3, img_shape[0] / 4, img_shape[0] * 3 / 4
    for line in lines:
        x1, y1, x2, y2 = line
        if (feature_zone[0] < x1 < feature_zone[1] and feature_zone[2] < y1 < feature_zone[3]) and \
                (feature_zone[0] < x2 < feature_zone[1] and feature_zone[2] < y2 < feature_zone[3]):
            line_count += 1

    print(f'目标区域直线数量：{line_count}')
    if line_count <= max_lines:
        return True

    else:
        return False


def defect2_hough(img_shape, lines, max_lines):
    """
    使用hough直线检测得到的直线判断图像是否有缺陷
    :param img_shape: 图像shape
    :param lines: 所有直线
    :param max_lines: 关注的区域内允许的最大直线数
    :return: True表示图像没有缺陷
    """
    line_count = 0
    half_shape = img_shape[0] / 2, img_shape[1] / 2
    for line in lines:
        x1, y1, x2, y2 = line
        if (x1 < half_shape[1] and y1 > half_shape[0]) and (x2 < half_shape[1] and y2 > half_shape[0]):
            line_count += 1

    print(f'目标区域直线数量：{line_count}')
    if line_count <= max_lines:
        return True

    else:
        return False
