import math
import os
import cv2
import numpy
import torch
from models.net import U2NET
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

class Tester(object):
    def __init__(self, is_cuda=True):
        self.net = U2NET(3, 2)
        self.device = torch.device('cuda:6' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load('weight/net.pt', map_location='cpu'), False)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval().to(self.device)
        '''以下为超参数，需要根据不同表盘类型设定'''
        self.line_width = 1600  # 表盘展开为直线的长度
        self.line_height = 150  # 表盘展开为直线的宽度
        self.circle_radius = 200  # 预设圆盘直df -h径
        self.circle_center = [208, 208]  # 圆盘指针的旋转中心
        # self.circle_center = [197, 226]  # 圆盘指针的旋转中心
        self.pi = 3.1415926535898

    @torch.no_grad()
    def __call__(self, image,image_path):
        image = self.square_picture(image, 416)
        cv2.imwrite(image_path,image)
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()
        point_mask = self.binary_image(mask[0])
        dail_mask = self.binary_image(mask[1])
        relative_value = self.get_relative_value(point_mask, dail_mask)
        print(relative_value)

        mask_out = np.zeros(image.shape, np.uint8)


        condition = point_mask ==1
        image[condition] = (0,0,255)
        mask_out[condition] = (0,0,255)
        condition = dail_mask == 1
        image[condition] = (0, 255, 0)
        mask_out[condition] = (0, 255, 0)
        #cv2.imwrite(image_path.replace('.jpg','__.png'),mask_out)
        cv2.imwrite(image_path.replace('.jpg','_.png'),image)
        # cv2.imshow('image_mask', mask_out)
        # cv2.waitKey()


    def binary_image(self, image):
        condition = image > 0.5
        image[condition] = 1
        image[~condition] = 0
        image = self.corrosion(image)
        return image

    # def make_border(self,image, top=16, bottom=0, left=11, right=0):
    #     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #     return image

    def get_relative_value(self, image_pointer, image_dail):
        # image_pointer = self.make_border(image_pointer,top=16, bottom=0, left=11, right=0)
        # image_dail = self.make_border(image_dail,top=16, bottom=0, left=11, right=0)

        line_image_dail = self.create_line_image(image_dail)
        line_image_pointer = self.create_line_image(image_pointer)

        data_1d_pointer = self.convert_1d_data(line_image_pointer)
        data_1d_dail = self.convert_1d_data(line_image_dail)
        data_1d_dail = self.mean_filtration(data_1d_dail)



        # cv2.imwrite('line_image_pointer.jpg', line_image_pointer)
        # cv2.imshow('line_image_dail', line_image_dail)
        '''定位指针相对刻度位置'''
        dail_flag = False
        pointer_flag = False
        one_dail_start = 0
        one_dail_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        dail_location = []
        pointer_location = 0
        for i in range(self.line_width - 1):
            if data_1d_dail[i] > 0 and data_1d_dail[i + 1] > 0:
                if not dail_flag:
                    one_dail_start = i
                    dail_flag = True
            if dail_flag:
                if data_1d_dail[i] == 0 and data_1d_dail[i + 1] == 0:
                    one_dail_end = i - 1
                    one_dail_location = (one_dail_start + one_dail_end) / 2
                    dail_location.append(one_dail_location)
                    one_dail_start = 0
                    one_dail_end = 0
                    dail_flag = False
            if data_1d_pointer[i] > 0 and data_1d_dail[i + 1] > 0:
                if not pointer_flag:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if data_1d_pointer[i] == 0 and data_1d_pointer[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False
        scale_num = len(dail_location)
        num_scale = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if dail_location[i] <= pointer_location < dail_location[i + 1]:
                    num_scale = i + (pointer_location - dail_location[i]) / (
                            dail_location[i + 1] - dail_location[i] + 1e-5) + 1
            ratio = (pointer_location - dail_location[0]) / (dail_location[-1] - dail_location[0] + 1e-5)
        result = {'scale_num': scale_num, 'num_sacle': num_scale, 'ratio': ratio}
        return result

    def create_line_image(self, image_mask):
        line_image = numpy.zeros((self.line_height, self.line_width), dtype=numpy.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                """计算与-y轴的夹角"""
                theta = ((2 * self.pi) / self.line_width) * (col + 1)
                '''计算当前扫描点位对应于原图的直径'''
                radius = self.circle_radius - row - 1
                '''计算当前扫描点对应于原图的位置'''
                y = int(self.circle_center[0] + radius * math.cos(theta) + 0.5)
                x = int(self.circle_center[1] - radius * math.sin(theta) + 0.5)

                # image_mask = self.make_border(image_mask, top=16, bottom=0, left=11, right=0)
                line_image[row, col] = image_mask[y, x]
        # line_image[line_image==1]=255
        # cv2.imshow('img',line_image)
        # cv2.waitKey(0)
        return line_image

    def convert_1d_data(self, line_image):
        """
        将图片转换为1维数组
        :param line_image: 展开的图片
        :return: 一维数组
        """
        data_1d = numpy.zeros((self.line_width), dtype=numpy.int16)
        for col in range(self.line_width):
            for row in range(self.line_height):
                if line_image[row, col] == 1:
                    data_1d[col] += 1


        return data_1d

    def corrosion(self, image):
        """
        腐蚀操作
        :param image:
        :return:
        """
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    def mean_filtration(self, data_1d_dail):
        """
        均值滤波
        :param data_1d_dail:
        :return:
        """
        mean_data = numpy.mean(data_1d_dail)
        for col in range(self.line_width):
            if data_1d_dail[col] < mean_data:
                data_1d_dail[col] = 0
        return data_1d_dail

    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len

        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

        h2, w2, _ = image.shape
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        # background = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        return background


if __name__ == '__main__':
    tester = Tester()
    # root = r'E:\data\diankeyuan\test\BigImg2\crop\yibiao'
    root = r'test'

    # for image_name in os.listdir(root):
    #     image_path = f'{root}/{image_name}'
    #     print(image_path)
    #     image = cv2.imread(image_path)
    #     tester(image,image_path)

    for rt, dir,files in os.walk(root):
        for image_name in files:
            image_path = f'{rt}/{image_name}'
            print(image_path)
            image = cv2.imread(image_path)
            tester(image, image_path)


    # image_path=r'E:\data\diankeyuan\segment\2_crop\biaoji_0_0.4/0001_base.jpg'
    # # image_path=r'E:\prj\ReadMeter-master\images/-01.jpg'
    # image = cv2.imread(image_path)
    # tester(image, image_path)




