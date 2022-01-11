"""
    这个文件的函数利用图像的特征判断图像是否有缺陷
"""


def correlation(ccoeff):
    if ccoeff < 0.8:
        return False

    else:
        return True


def region_area(area, normal_area):
    if float(area) / normal_area < 0.9:
        return False

    else:
        return True


def defect1_hough(img_shape, lines, max_lines):
    line_count = 0
    feature_zone = img_shape[1] / 3, img_shape[1] * 2 / 3, img_shape[0] / 4, img_shape[0] * 3 / 4
    for line in lines:
        x1, y1, x2, y2 = line
        if (feature_zone[0] < x1 < feature_zone[1] and feature_zone[2] < y1 < feature_zone[3]) and \
                (feature_zone[0] < x2 < feature_zone[1] and feature_zone[2] < y2 < feature_zone[3]):
            line_count += 1

    print(line_count)
    if line_count <= max_lines:
        return True

    else:
        return False


def defect2_hough(img_shape, lines, max_lines):
    line_count = 0
    half_shape = img_shape[0] / 2, img_shape[1] / 2
    for line in lines:
        x1, y1, x2, y2 = line
        if (x1 < half_shape[1] and y1 > half_shape[0]) and (x2 < half_shape[1] and y2 > half_shape[0]):
            line_count += 1

    if line_count <= max_lines:
        return True

    else:
        return False
