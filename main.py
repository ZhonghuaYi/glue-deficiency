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


def template_match(image, templates, canny):
    start_time = time.time()  # 设定程序开始运行时间
    f = "ccoeff"  # 用来分类的特征

    # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.4, 1)
    # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    t_count = 0
    for template in templates:
        result = None
        if f == "ccoeff":
            # 检测
            CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count])

            print(f"区域{t_count+1}相关系数：{CCOEFF}")
            result = feature.correlation(CCOEFF, 0.4, 0.2)

        elif f == "sift":
            # 检测
            CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count])
            # 创建sift实例
            sift = cv.SIFT_create()
            # 模板1的sift特征
            kp, des = sift.detectAndCompute(template, None)
            template_sift = cv.drawKeypoints(template, kp, None)
            cv.imshow(f'template{t_count+1}_sift', template_sift)
            # 图像的roi的sift特征
            kp_img, des_img = sift.detectAndCompute(image_roi, None)
            img_sift = cv.drawKeypoints(image_roi, kp_img, None)
            cv.imshow(f'roi{t_count+1}_sift', img_sift)
            # 将图像1和模板1的sift特征进行匹配
            bf = cv.BFMatcher(crossCheck=True)
            if kp_img:
                matchs = bf.match(des, des_img)
                match_image = cv.drawMatches(template, kp, image, kp_img, matchs, None, flags=2)
                cv.imshow(f'match1', match_image)
            else:
                matchs = []
            result = feature.key_points(kp, matchs, 0.7, 0.5, 100)
        elif f == "hausdorff":
            image_roi = roi.hausdorff_match(template, image.copy(), canny[t_count])
            cv.imshow(f"roi{t_count+1}", image_roi)

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

        result_explain(result, t_count+1)
        t_count += 1

    end_time = time.time()  # 记录程序结束运行时间
    print('运行时间：', end_time - start_time, "s")
    print("————————————")
    cv.waitKey(0)


if __name__ == '__main__':
    sample_set = 2
    sample = None
    templates = []
    canny = []
    if sample_set == 1:
        # 读取样本
        sample_root = "./image/sample/"
        sample = sample_generate(sample_root)
        # defect1()  # 检测第一种缺陷
        # defect2()  # 检测第二种缺陷

        # 读取参考样本
        refer1_root = "./image/refer1/"
        refer1_sample = refer_generate(refer1_root)
        refer2_root = "./image/refer2/"
        refer2_sample = refer_generate(refer2_root)

        canny1 = (50, 100)
        canny2 = (100, 200)
        canny = [canny1, canny2]  # canny法的两个阈值
        # 生成模板
        template1 = template_generate(refer1_sample, x=(50, 300), y=(50, 300), canny=canny1)
        templates.append(template1)
        template2 = template_generate(refer2_sample, x=(20, 100), y=(220, 470), canny=canny2)
        templates.append(template2)
        # for i in range(len(templates)):
        #     cv.imshow(f"template{i+1}", templates[i])
        # cv.waitKey(0)

        # # 读取模板图像
        # template1_path = refer1_root + 'target_template.BMP'
        # template1 = cv.imread(template1_path, 0)
        # template2_path = refer2_root + 'target_template.BMP'
        # template2 = cv.imread(template2_path, 0)

    elif sample_set == 2:
        # 读取样本
        sample_root = "./image2/sample/"
        sample = []
        for img in sample_generate(sample_root, flag=1):
            sample.append(img[:, :, 2])

        # 读取参考样本
        refer_root = "./image2/refer/"
        refer_sample = []
        for img in refer_generate(refer_root, flag=1):
            refer_sample.append(img[:, :, 2])

        canny1 = (0, 200)
        canny = [canny1, canny1, canny1, canny1, canny1]
        # 生成模板
        # template = template_generate(refer_sample, x=(0, -1), y=(0, -1), canny=canny1)
        # cv.imshow("template", template)
        template1 = template_generate(refer_sample, x=(150, 300), y=(120, 270), canny=canny1)
        templates.append(template1)
        template2 = template_generate(refer_sample, x=(170, 270), y=(230, 330), canny=canny1)
        templates.append(template2)
        template3 = template_generate(refer_sample, x=(70, 150), y=(300, 400), canny=canny1)
        templates.append(template3)
        template4 = template_generate(refer_sample, x=(250, 350), y=(250, 350), canny=canny1)
        templates.append(template4)
        template5 = template_generate(refer_sample, x=(300, 400), y=(100, 250), canny=canny1)
        templates.append(template5)
        # for i in range(len(templates)):
        #     cv.imshow(f"template{i+1}", templates[i])
        # cv.waitKey(0)

    count = 1
    for image in sample:
        print(f"样本{count}：")
        template_match(image, templates, canny)
        count += 1
