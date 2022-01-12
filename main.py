import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from func import *
import segment
import feature


def defect1():
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

    method = "template_match"  # 使用的分割方法
    f = "ccoeff"  # 用来分类的特征

    if method == 'thresh':
        area_percent = 0.3
        pre_area_num = 12
        normal_area = 420

        # 用于形态学计算的矩形结构元素
        structure_element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

        for image in sample:
            region_area, image = segment.threshold_segment(image, area_percent, pre_area_num, structure_element)

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)
            count += 1  # 图像计数加一

            # 根据特征判断此样本是否合格（合格为True）
            result = feature.region_area(region_area, normal_area)
            results.append(result)

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print("运行时间：", end_time - start_time, "s")
        cv.waitKey(0)

    elif method == 'template_match':
        canny = (50, 120)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(50, 300), y=(50, 300), canny=canny)
        # cv.imshow('img', target_template)

        # # 读取模板图像
        # template_path = image_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.3, 1)
            # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            max_ccoeff = 0
            match_image = None
            for i in range(0, 20):
                angle = i * 0.5 - 5
                M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                temp_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

                CCOEFF, temp_image = segment.template_match(temp_image, target_template, canny)
                if CCOEFF >= max_ccoeff:
                    max_ccoeff = CCOEFF
                    match_image = temp_image

            image = match_image

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if f == "ccoeff":
                print(f"样本{count}相关系数：{max_ccoeff}")
                result = feature.correlation(max_ccoeff)
                results.append(result)

            elif f == "hough":
                drawing = np.zeros(image.shape, dtype=np.uint8)
                lines = defect1_hough_line(image)
                drawing = draw_line(drawing, lines)
                cv.imshow('hough' + str(count), drawing)
                result = feature.defect1_hough(image.shape, lines, 2)
                results.append(result)

            count += 1  # 图像计数加一

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print('运行时间：', end_time - start_time, "s")
        cv.waitKey(0)


def defect2():
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

    method = "template_match"  # 使用的分割方法
    f = "ccoeff"  # 用来分类的特征

    if method == 'template_match':
        canny = (100, 200)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(20, 100), y=(220, 470), canny=canny)
        cv.imshow("template", target_template)

        # # 读取模板图像
        # template_path = image_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.3, 1)
            # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            max_ccoeff = 0
            match_image = None
            for i in range(0, 20):
                angle = i*0.5 - 5
                M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                temp_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

                CCOEFF, temp_image = segment.template_match(temp_image, target_template, canny)
                if CCOEFF >= max_ccoeff:
                    max_ccoeff = CCOEFF
                    match_image = temp_image

            image = match_image

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if f == "ccoeff":
                print(f"样本{count}相关系数：{max_ccoeff}")
                result = feature.correlation(max_ccoeff)
                results.append(result)

            elif f == "hough":
                drawing = np.zeros(image.shape, dtype=np.uint8)
                lines = defect2_hough_line(image)
                drawing = draw_line(drawing, lines)
                cv.imshow('hough'+str(count), drawing)
                result = feature.defect2_hough(image.shape, lines, 10)
                results.append(result)

            count += 1  # 图像计数加一

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print('运行时间：', end_time - start_time, "s")
        cv.waitKey(0)


if __name__ == '__main__':
    # defect1()  # 检测第一种缺陷
    defect2()  # 检测第二种缺陷
