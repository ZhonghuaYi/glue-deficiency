import time

import data
from data import *
import func
from func import *
import roi
import feature


def thresh_segment(image, area_percent, structure_element, normal_area, thresh):
    start_time = time.time()  # 设定程序开始运行时间
    
    image = image[0:800, 0:800]
    region_area, image = roi.threshold_segment(image, area_percent, structure_element)

    # 显示最终分离出的区域的图像
    cv.imshow("image", image)

    # 根据特征判断此样本是否合格（合格为True）
    result = feature.region_area(region_area, normal_area, thresh)

    print("判断结果：", result)
    end_time = time.time()  # 记录程序结束运行时间
    print("运行时间：", end_time - start_time, "s")
    cv.waitKey(0)


def template_match(image, edge_templates, templates, canny, f, thresh):
    start_time = time.time()  # 设定程序开始运行时间

    # M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 1.4, 1)
    # image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    t_count = 0
    for template in edge_templates:
        result = None
        if f == "ccoeff":
            # 检测
            CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count])
            # cv.imshow(f"img{t_count}", image_roi)
            print(f"区域{t_count+1}相关系数：{CCOEFF}")
            result = feature.correlation(CCOEFF, thresh[1], thresh[0])

        elif f == "sift":
            USE_DES = True
            if USE_DES:
                # 检测
                CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count], flag=1)
                # 创建sift实例
                sift = cv.SIFT_create()
                # 模板的sift特征
                kp_t, des_t = sift.detectAndCompute(templates[t_count], None)
                template_sift = cv.drawKeypoints(templates[t_count], kp_t, None)
                cv.imshow(f'template{t_count+1}_sift', template_sift)
                # 图像的roi的sift特征
                kp_img, des_img = sift.detectAndCompute(image_roi, None)
                img_sift = cv.drawKeypoints(image_roi, kp_img, None)
                cv.imshow(f'roi{t_count+1}_sift', img_sift)
                # 将图像和模板的sift特征进行匹配
                bf = cv.BFMatcher(crossCheck=True)
                if kp_img:
                    matchs = bf.match(des_t, des_img)
                    new_kp_t = []
                    new_kp_img = []
                    for m in matchs:
                        new_kp_t.append(kp_t[m.queryIdx])
                        new_kp_img.append(kp_img[m.trainIdx])
                    matchs = key_point_match(new_kp_t, new_kp_img, 10)
                    kp_t = new_kp_t
                    kp_img = new_kp_img
                    match_image = cv.drawMatches(templates[t_count], new_kp_t, image_roi, new_kp_img, matchs, None, flags=2)
                    cv.imshow(f'match{t_count+1}', match_image)
                else:
                    matchs = []
            else:
                # 检测
                CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count], flag=1)
                # 创建sift实例
                sift = cv.SIFT_create()
                # 获取特征点
                kp_t = sift.detect(templates[t_count], None)
                kp_img = sift.detect(image_roi, None)
                # 计算位置相近的特征点
                if kp_img:
                    matchs = key_point_match(kp_t, kp_img, 10)
                    match_image = cv.drawMatches(templates[t_count], kp_t, image_roi, kp_img, matchs, None, flags=2)
                    cv.imshow(f'match{t_count + 1}', match_image)
                else:
                    matchs = []

            result = feature.key_points(kp_t, kp_img, matchs, thresh[1], thresh[0], 1000)

        # 霍夫不太适合不同缺陷的同时检测
        elif f == "hough":
            # 检测
            CCOEFF, image_roi = roi.template_match(image.copy(), template, canny[t_count])
            if t_count == 0:
                drawing = np.zeros(image_roi.shape, dtype=np.uint8)
                lines = defect1_hough_line(image_roi)
                drawing = draw_line(drawing, lines)
                cv.imshow('hough1', drawing)
                result = feature.defect1_hough(image_roi.shape, lines, 4)
            elif t_count == 1:
                drawing = np.zeros(image_roi.shape, dtype=np.uint8)
                lines = defect2_hough_line(image_roi)
                drawing = draw_line(drawing, lines)
                cv.imshow('hough2', drawing)
                result = feature.defect2_hough(image_roi.shape, lines, 10)

        result_explain(result, t_count+1)
        t_count += 1

    end_time = time.time()  # 记录程序结束运行时间
    print('运行时间：', end_time - start_time, "s")
    print("————————————")
    cv.waitKey(0)


def sift_match(image, templates):
    start_time = time.time()  # 设定程序开始运行时间

    t_count = 0
    image = func.image_resize(image, 500)
    image = cv.medianBlur(image, 5)
    for template in templates:
        result = None
        # 创建sift实例
        sift = cv.SIFT_create()
        # 模板的sift特征
        kp_t, des_t = sift.detectAndCompute(template, None)
        # template_sift = cv.drawKeypoints(template, kp_t, None)
        # cv.imshow(f'template{t_count + 1}_sift', template_sift)
        # 图像的的sift特征
        kp_img, des_img = sift.detectAndCompute(image, None)
        # img_sift = cv.drawKeypoints(image, kp_img, None)
        # cv.imshow(f'img{t_count + 1}_sift', img_sift)
        # 设置FLANN匹配器参数，定义FLANN匹配器，使用 KNN 算法实现匹配
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # 将图像和模板的sift特征进行匹配
        flann = cv.FlannBasedMatcher(index_params, search_params)
        if kp_img:
            matches = flann.knnMatch(des_t, des_img, k=2)
            # 根据matches生成相同长度的matchesMask列表，列表元素为[0,0]
            matches_mask = [[0, 0] for i in range(len(matches))]
            # 去除错误匹配
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matches_mask[i] = [1, 0]
            # 将图像显示
            # matchColor是两图的匹配连接线，连接线与matchesMask相关
            # singlePointColor是勾画关键点
            drawParams = dict(matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              matchesMask=matches_mask[:50],
                              flags=0)
            result_image = cv.drawMatchesKnn(template, kp_t, image, kp_img, matches[:50], None, **drawParams)
            cv.imshow(f'match{t_count + 1}', result_image)
        else:
            matches = []
            result = 2

        result_explain(result, t_count + 1)
        t_count += 1

    end_time = time.time()  # 记录程序结束运行时间
    print('运行时间：', end_time - start_time, "s")
    print("————————————")
    cv.waitKey(0)


if __name__ == '__main__':
    data = data.DataLoader()
    data.load(sample_set=1, segment="template_match", f="sift")
    count = 1
    for image in data.sample:
        print(f"样本{count}：")
        thresh_segment(image, data.area_percent, data.structure_element, data.normal_area, data.thresh)
        # template_match(image, data.edge_templates, data.templates, data.canny, data.f, thresh)
        # sift_match(image, data.templates)
        count += 1

    cv.waitKey(0)
    cv.destroyAllWindows()
