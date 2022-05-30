from func import *


class DataLoader:
    def __init__(self):
        self.f = None
        self.thresh = None
        self.canny = None
        self.templates = None
        self.edge_templates = None
        self.structure_element = None
        self.normal_area = None
        self.area_percent = None
        self.sample = None

    def load(self, sample_set, segment, f=None):
        if sample_set == 1:

            # 样本图片的位置
            sample_root = "./image/sample/"
            refer1_root = "./image/refer1/"
            refer2_root = "./image/refer2/"

            # 读取样本
            self.sample = sample_generate(sample_root)

            # 读取参考样本
            refer1_sample = refer_generate(refer1_root)
            refer2_sample = refer_generate(refer2_root)

            if segment == "thresh_segment":
                self.area_percent = 0.3
                self.normal_area = 420
                self.thresh = 0.9

                # 用于形态学计算的矩形结构元素
                self.structure_element = cv.getStructuringElement(
                    cv.MORPH_RECT, (7, 7))

            elif segment == "template_match":
                self.edge_templates = []
                self.templates = []

                canny1 = (50, 100)
                canny2 = (100, 200)  # 500x下的canny
                self.canny = [canny1, canny2]  # canny法的两个阈值

                self.f = f
                self.thresh = (0.1, 0.75)

                # 生成模板
                template1, edge_template1 = template_generate(refer1_sample,
                                                              x=(50, 300),
                                                              y=(50, 300),
                                                              canny=canny1)
                template2, edge_template2 = template_generate(refer2_sample,
                                                              x=(20, 100),
                                                              y=(220, 470),
                                                              canny=canny2)
                self.edge_templates.append(edge_template1)
                self.edge_templates.append(edge_template2)
                self.templates.append(template1)
                self.templates.append(template2)
                # for i in range(len(templates)):
                #     cv.imshow(f"template{i+1}", templates[i])
                # cv.waitKey(0)

                # # 读取模板图像
                # template1_path = refer1_root + 'target_template.BMP'
                # template1 = cv.imread(template1_path, 0)
                # template2_path = refer2_root + 'target_template.BMP'
                # template2 = cv.imread(template2_path, 0)
