"""
    这个文件的函数利用图像的特征判断图像是否有缺陷
"""


def correlation(ccoeff):
    if ccoeff < 0.9:
        return False

    else:
        return True


def region_area(area, normal_area):
    if float(area) / normal_area < 0.9:
        return False

    else:
        return True
