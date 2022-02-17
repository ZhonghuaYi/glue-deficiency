import time
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from func import *
import roi
import feature


def defect1():
    # 设定程序开始运行时间
    start_time = time.time()

    # 读取参考样本
    refer_root = "./image/refer1/"
    refer_sample = refer_generate(refer_root)

    # 读取样本
    sample_root = "./image/sample/"
    sample_list = ["sample000.BMP", "sample001.BMP", "sample002.BMP"]
    sample = sample_generate(sample_root, sample_list)

    count = 1  # 图像的计数

    results = []  # 判断结果

    method = "template_match"  # 使用的分割方法
    f = "hough"  # 用来分类的特征

    if method == 'thresh':
        area_percent = 0.3
        pre_area_num = 12
        normal_area = 420

        # 用于形态学计算的矩形结构元素
        structure_element = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

        for image in sample:
            region_area, image = roi.threshold_segment(image, area_percent, pre_area_num, structure_element)

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
        canny = (50, 100)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(50, 300), y=(50, 300), canny=canny)
        # cv.imshow('template', target_template)

        # # 读取模板图像
        # template_path = refer_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.4, 1)
            # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            CCOEFF, image = roi.template_match(image, target_template, canny)

            # 显示最终分离出的区域的图像
            # window_name = 'img' + str(count)  # 图像窗口的名称
            # cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if f == "ccoeff":
                print(f"样本{count}相关系数：{CCOEFF}")
                result = feature.correlation(CCOEFF, 0.6, 0.2)
                results.append(result)

            elif f == "hough":
                drawing = np.zeros(image.shape, dtype=np.uint8)
                lines = defect1_hough_line(image)
                drawing = draw_line(drawing, lines)
                cv.imshow('hough' + str(count), drawing)
                result = feature.defect1_hough(image.shape, lines, 10)
                results.append(result)

            count += 1  # 图像计数加一

        print("判断结果：", results)
        end_time = time.time()  # 记录程序结束运行时间
        print('运行时间：', end_time - start_time, "s")
        cv.waitKey(0)

    elif method == "sift":
        canny = (50, 100)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(50, 300), y=(50, 300), canny=canny)
        # cv.imshow('template', target_template)

        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(target_template, None)

        for image in sample:
            M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.3, 1)
            image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # 第一步，将图像缩放到一个统一的大小（较小边为500像素）
            scale = min(image.shape) / 500
            new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
            image = cv.resize(image, new_size)

            # 第二步，对图像进行高斯平滑
            image = cv.GaussianBlur(image, (3, 3), sigmaX=1)

            # 第三步，Canny法提取图像边缘
            image = cv.Canny(image, canny[0], canny[1])

            # 图像的sift特征
            kp_img, des_img = sift.detectAndCompute(image, None)

            # 将图像和模板的特征进行匹配
            bf = cv.BFMatcher()
            matchs = bf.match(des, des_img)
            match_image = cv.drawMatches(target_template, kp, image, kp_img, matchs, None, flags=2)
            cv.imshow(f'img{count}_sift', match_image)
            count += 1

        end_time = time.time()  # 记录程序结束运行时间
        print("运行时间：", end_time - start_time, "s")
        cv.waitKey(0)


def defect2():
    # 设定程序开始运行时间
    start_time = time.time()

    # 读取参考样本
    refer_root = "./image/refer2/"
    refer_sample = refer_generate(refer_root)

    # 读取样本
    sample_root = "./image/sample/"
    sample_list = ["sample003.BMP", "sample004.BMP"]
    sample = sample_generate(sample_root, sample_list)

    count = 1  # 图像的计数

    results = []  # 判断结果

    method = "sift"  # 使用的分割方法
    f = "ccoeff"  # 用来分类的特征

    if method == 'template_match':
        canny = (100, 200)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(20, 100), y=(220, 470), canny=canny)
        cv.imshow("template", target_template)

        # # 读取模板图像
        # template_path = refer_root + 'target_template.BMP'
        # target_template = cv.imread(template_path, 0)

        for image in sample:
            M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.4, 1)
            image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            CCOEFF, image = roi.template_match(image, target_template, canny)

            # 显示最终分离出的区域的图像
            window_name = 'img' + str(count)  # 图像窗口的名称
            cv.imshow(window_name, image)

            # 根据特征判断此样本是否合格（合格为True）
            if f == "ccoeff":
                print(f"样本{count}相关系数：{CCOEFF}")
                result = feature.correlation(CCOEFF, 0.6, 0.2)
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

    elif method == "sift":
        canny = (100, 200)  # canny法的两个阈值

        # 生成模板
        target_template = template_generate(refer_sample, x=(20, 100), y=(220, 470), canny=canny)
        # cv.imshow('template', target_template)

        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(target_template, None)
        template_sift = cv.drawKeypoints(target_template, kp, None)
        cv.imshow('template_sift', template_sift)

        for image in sample:
            # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.3, 1)
            # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # 第一步，将图像缩放到一个统一的大小（较小边为500像素）
            scale = min(image.shape) / 500
            new_size = round(image.shape[1] / scale), round(image.shape[0] / scale)  # 这里的size指宽度和高度
            image = cv.resize(image, new_size)

            # # 第二步，对图像进行高斯平滑
            image = cv.GaussianBlur(image, (3, 3), sigmaX=1)

            # 第三步，Canny法提取图像边缘
            image = cv.Canny(image, canny[0], canny[1])

            # 图像的sift特征
            kp_img, des_img = sift.detectAndCompute(image, None)
            img_sift = cv.drawKeypoints(image, kp_img, None)
            cv.imshow(f'img{count}', img_sift)

            # 将图像和模板的特征进行匹配
            bf = cv.BFMatcher()
            matchs = bf.match(des, des_img)
            match_image = cv.drawMatches(target_template, kp, image, kp_img, matchs, None, flags=2)
            cv.imshow(f'img{count}_match', match_image)
            count += 1

        end_time = time.time()  # 记录程序结束运行时间
        print("运行时间：", end_time - start_time, "s")
        cv.waitKey(0)


def template_match(image):
    # 设定程序开始运行时间
    start_time = time.time()

    # 读取参考样本
    refer1_root = "./image/refer1/"
    refer1_sample = refer_generate(refer1_root)
    refer2_root = "./image/refer2/"
    refer2_sample = refer_generate(refer2_root)

    result1 = None
    result2 = None# 判断结果
    f = "hausdorff"  # 用来分类的特征
    canny1 = (50, 100)
    canny2 = (100, 200)  # canny法的两个阈值

    # 生成模板
    template1 = template_generate(refer1_sample, x=(50, 300), y=(50, 300), canny=canny1)
    # cv.imshow("template1", template1)
    template2 = template_generate(refer2_sample, x=(20, 100), y=(220, 470), canny=canny2)
    # cv.imshow("template2", template2)

    # # 读取模板图像
    # template1_path = refer1_root + 'target_template.BMP'
    # template1 = cv.imread(template1_path, 0)
    # template2_path = refer2_root + 'target_template.BMP'
    # template2 = cv.imread(template2_path, 0)

    # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.4, 1)
    # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    if f == "ccoeff":
        # 检测
        CCOEFF1, image1 = roi.template_match(image.copy(), template1, canny1)  # 检测第一种缺陷
        CCOEFF2, image2 = roi.template_match(image.copy(), template2, canny2)  # 检测第二种缺陷

        print(f"样本相关系数：{CCOEFF1} {CCOEFF2}")
        result1 = feature.correlation(CCOEFF1, 0.6, 0.2)
        result2 = feature.correlation(CCOEFF2, 0.6, 0.2)

    elif f == "sift":
        # 检测
        CCOEFF1, image1 = roi.template_match(image.copy(), template1, canny1)  # 检测第一种缺陷
        CCOEFF2, image2 = roi.template_match(image.copy(), template2, canny2)  # 检测第二种缺陷
        # 创建sift实例
        sift = cv.SIFT_create()
        # 模板1的sift特征
        kp1, des1 = sift.detectAndCompute(template1, None)
        template_sift = cv.drawKeypoints(template1, kp1, None)
        cv.imshow('template1_sift', template_sift)
        # 模板2的sift特征
        kp2, des2 = sift.detectAndCompute(template2, None)
        template_sift = cv.drawKeypoints(template2, kp2, None)
        cv.imshow('template2_sift', template_sift)
        # 图像1的sift特征
        kp_img1, des_img1 = sift.detectAndCompute(image1, None)
        img_sift = cv.drawKeypoints(image1, kp_img1, None)
        cv.imshow(f'img1_sift', img_sift)
        # 图像2的sift特征
        kp_img2, des_img2 = sift.detectAndCompute(image2, None)
        img_sift = cv.drawKeypoints(image2, kp_img2, None)
        cv.imshow(f'img2_sift', img_sift)
        # 将图像1和模板1的sift特征进行匹配
        bf = cv.BFMatcher(crossCheck=True)
        if kp_img1:
            matchs = bf.match(des1, des_img1)
            match_image = cv.drawMatches(template1, kp1, image1, kp_img1, matchs, None, flags=2)
            cv.imshow(f'match1', match_image)
        else:
            matchs = []
        result1 = feature.key_points(kp1, matchs, 0.7, 0.5, 100)
        # 将图像2和模板2的sift特征进行匹配
        if kp_img2:
            matchs = bf.match(des2, des_img2)
            match_image = cv.drawMatches(template2, kp2, image2, kp_img2, matchs, None, flags=2)
            cv.imshow(f'match2', match_image)
        else:
            matchs = []
        result2 = feature.key_points(kp2, matchs, 0.7, 0.5, 100)

    elif f == "hausdorff":
        image1 = roi.hausdorff_match(template1, image.copy(), canny1)
        cv.imshow("img1", image1)
        image2 = roi.hausdorff_match(template2, image.copy(), canny2)
        cv.imshow("img2", image2)

    # 霍夫不太适合不同缺陷的同时检测
    # elif f == "hough":
    #     drawing = np.zeros(image.shape, dtype=np.uint8)
    #     lines = defect1_hough_line(image)
    #     drawing = draw_line(drawing, lines)
    #     cv.imshow('hough1', drawing)
    #     result1 = feature.defect1_hough(image.shape, lines, 10)
    #
    #     drawing = np.zeros(image.shape, dtype=np.uint8)
    #     lines = defect2_hough_line(image)
    #     drawing = draw_line(drawing, lines)
    #     cv.imshow('hough2', drawing)
    #     result2 = feature.defect2_hough(image.shape, lines, 10)

    result_explain(result1, 1)
    result_explain(result2, 2)
    end_time = time.time()  # 记录程序结束运行时间
    print('运行时间：', end_time - start_time, "s")
    cv.waitKey(0)


if __name__ == '__main__':
    sample_root = "./image/sample/"
    sample = sample_generate(sample_root)
    # defect1()  # 检测第一种缺陷
    # defect2()  # 检测第二种缺陷
    count = 1
    for image in sample:
        print(f"image{count}")
        template_match(image)
        print("————————————")
        count += 1
