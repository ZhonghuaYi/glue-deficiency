import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from func import *
import segment
import feature


def class1_defect1(**method):
    # 设定程序开始运行时间
    start_time = time.time()

    # 读取样本
    image_root = './image/class1_defect1/'
    refer_images = ['refer000.BMP']  # 参考图像
    defect_images = ['defect000.BMP', 'defect001.BMP']  # 缺陷图像
    refer_list = [image_root + image for image in refer_images]
    defect_list = [image_root + image for image in defect_images]
    refer_sample = refer_sample_generate(refer_list)  # 参考样本的生成器
    sample = sample_generate(refer_list, defect_list)  # 样本的生成器

    count = 1  # 图像的计数

    results = []  # 判断结果

    if method["segment"] == 'threshold_segment':
        # 用于形态学计算的矩形结构元素
        structure_element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

        for image in sample:
            region_area, image = segment.threshold_segment(image, method["area_percent"], method["pre_area_num"],
                                                           structure_element)

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)
            count += 1  # 图像计数加一

            # 根据特征判断此样本是否合格（合格为True）
            result = feature.region_area(region_area, method["normal_area"])
            results.append(result)

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print("运行时间：", end_time - start_time, "s")
        cv.waitKey(0)

    elif method["segment"] == 'template_match':
        # 生成模板
        target_template = template_generate(refer_sample, x=(50, 300), y=(50, 300), canny=method["canny"])
        # cv.imshow('img', target_template)

        # # 读取模板图像
        # template_path = image_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            # 使用模板匹配的方式提取出目标区域
            CCOEFF, image = segment.template_match(image, target_template, method["canny"])

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if method["feature"] == "ccoeff":
                print(f"样本{count}相关系数：{CCOEFF}")
                result = feature.correlation(CCOEFF)
                results.append(result)

            count += 1  # 图像计数加一

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print('运行时间：', end_time - start_time, "s")
        cv.waitKey(0)


def class1_defect2(**method):
    # 设定程序开始运行时间
    start_time = time.time()

    # 读取样本
    image_root = './image/class1_defect2/'
    refer_images = ['refer000.BMP']
    defect_images = ['defect000.BMP']
    refer_list = [image_root + image for image in refer_images]
    defect_list = [image_root + image for image in defect_images]
    refer_sample = refer_sample_generate(refer_list)  # 参考样本的生成器
    sample = sample_generate(refer_list, defect_list)  # 样本的生成器

    count = 1  # 图像的计数

    results = []  # 判断结果

    if method["segment"] == 'template_match':
        # 生成模板
        target_template = template_generate(refer_sample, x=(20, 100), y=(220, 470), canny=method["canny"])
        # cv.imshow('img', target_template)

        # # 读取模板图像
        # template_path = image_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            # 使用模板匹配的方式提取出目标区域
            CCOEFF, image = segment.template_match(image, target_template, method["canny"])

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if method["feature"] == "ccoeff":
                print(f"样本{count}相关系数：{CCOEFF}")
                result = feature.correlation(CCOEFF)
                results.append(result)

            count += 1  # 图像计数加一

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print('运行时间：', end_time - start_time, "s")
        cv.waitKey(0)


if __name__ == '__main__':
    threshold_segment_method = {
        "segment": "threshold_segment",
        "area_percent": 0.3,
        "pre_area_num": 12,
        "normal_area": 420,
    }
    template_match_method = {
        "segment": "template_match",
        "canny": (50, 120),
        "feature": "ccoeff",
    }
    class1_defect1(**template_match_method)
