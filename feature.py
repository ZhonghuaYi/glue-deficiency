"""
    这个文件的函数利用图像的特征判断图像是否有缺陷
"""


def correlation(ccoeff, th, th0):
    """
    用相关系数判断图像是否有缺陷
    :param ccoeff: 相关系数
    :param th: 相关系数的阈值，小于此值认为可能有缺陷
    :param th0: 相关系数的阈值，小于此值认为没有匹配区域
    :return: 1表示图像没有缺陷，0表示图像存在缺陷，2表示未检测到目标区域
    """
    if ccoeff < th0:
        return 2
    elif th0 <= ccoeff < th:
        return 0
    else:
        return 1


def region_area(area, normal_area, thresh):
    """
    用检测到的区域与没有缺陷时的区域面积比值判断图像是否有缺陷
    :param area: 检测到的区域面积
    :param normal_area: 没有缺陷时的区域的面积
    :return: True表示图像没有缺陷
    """
    if float(area) / normal_area < thresh:
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
    # 设定关注的区域范围
    feature_zone = img_shape[1] / 3, img_shape[1] * 2 / 3, img_shape[
        0] / 4, img_shape[0] * 3 / 4
    # 当直线位于关注的区域范围内时，累计一条直线
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
        if (x1 < half_shape[1]
                and y1 > half_shape[0]) and (x2 < half_shape[1]
                                             and y2 > half_shape[0]):
            line_count += 1

    print(f'目标区域直线数量：{line_count}')
    if line_count <= max_lines:
        return True

    else:
        return False


def key_points(t_kp, t_img, matches, th, th0, min_distance):
    """
    根据特征点的匹配情况返回判断结果
    :param t_kp: 模板的特征点
    :param t_img: 检测图像的特征点
    :param matches: 模板与图像匹配的结果
    :param th: 匹配数量与模板特征点数量比值的阈值，小于此值认为可能有缺陷
    :param th0: 匹配数量与模板特征点数量比值的阈值，小于此值认为没有匹配区域
    :param min_distance: 匹配距离的平均值的阈值
    :return: 1表示图像没有缺陷，0表示图像存在缺陷，2表示未检测到目标区域
    """
    match_percent = len(matches) / (len(t_kp) + len(t_img) - len(matches))
    print(f"match percent: {match_percent}")
    if match_percent < th0:
        return 2
    elif th0 <= match_percent < th:
        return 0
    else:
        distance_sum = 0
        for m in matches:
            distance_sum += m.distance
        average_distance = distance_sum / len(matches)
        if average_distance < min_distance:
            return 1
        else:
            return 0
