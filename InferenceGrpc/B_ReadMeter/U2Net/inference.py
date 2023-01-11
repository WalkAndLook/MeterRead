'''
用PCA降维去拟合直线
'''

import sys
import cv2
import time
import numpy
import torch
import math
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model.net import U2NET

class meterPoint(object):

    def __init__(self, is_cuda=True):
        self.net = U2NET(3, 2)  # 输出2个mask
        self.device = torch.device('cuda:1' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load(str(ROOT) + '/weight/net_meter.pt', map_location='cpu'), False)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
        self.net.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, image):

        # 图片放缩到416*416 是32的倍数
        image_416 = self.square_picture(image, 416)
        # 更改图片通道顺序
        image_tensor = self.to_tensor(image_416.copy()).to(self.device)
        # 送入模型
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        # 降维 torch-4变成了numpy-3
        mask = d0.squeeze(0).cpu().numpy()
        meter_mask = self.binary_image(mask[0])
        meter = np.zeros((416, 416, 3), dtype='uint8')
        condition = meter_mask == 1
        meter[condition] = (0, 0, 255)
        image_416[condition] = (0, 0, 255)
        first_point, second_point, center_pointer = self.fitline(meter, image_416)
        first_point = self.position_return_image(image, first_point)
        second_point = self.position_return_image(image, second_point)
        center_pointer = self.position_return_image(image, center_pointer)
        # 计算两个点到重心直接的距离，确保离针尖进的点为第一个点
        dist1 = self.point_dist(first_point, center_pointer)
        dist2 = self.point_dist(second_point, center_pointer)
        if dist1 <= dist2:
            first_point, second_point = second_point, first_point
        else:
            pass

        # 计算从指针靠近底部的点到靠近针尖的点的向量, 因为后面还需要计算没裁剪前的位置，以及后面计算读数时需要变换坐标系，
        # 这里先不计算指针向量，只返回第一个点和第二个点
        point_vector = [first_point[0] - second_point[0], first_point[1] - second_point[1]]
        return point_vector

    @staticmethod
    def point_dist(point1, point2):
        '''

        :param point1: 第一个点，（x，y），可以是列表，也可以是元组
        :param point2: 第二个点
        :return: 两个点直接的距离，常量float
        '''
        return math.sqrt(pow((point1[0] - point2[0]), 2) + pow((point1[1] - point2[1]), 2))

    def position_return_image(self, image, pointer):
        h, w = image.shape[0:2]
        fx = 416 / w
        fy = 416 / h
        x = pointer[0]
        y = pointer[1]
        if w >= h:
            x = x / fx
            y = (y - ((416 - h * fx)/2)) / fx
        else:
            x = (x - ((416 - h * fy)/2)) / fy
            y = y / fy
        # 取整
        x = int(x + 0.5)
        y = int(y + 0.5)
        return [x, y]

    def fitline(self, mask, image):
        # 对图像进行灰度处理
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 返回mask坐标
        # np.where 返回一个元组，第一个是x坐标，第二个是y坐标
        xy = np.where(gray_mask > 38)
        # xy (array([217, 218, 218, ..., 436, 436, 437], dtype=int64), array([202, 201, 202, ..., 417, 418, 417], dtype=int64))
        x = np.array((xy)[0])
        y = np.array((xy)[1])
        length = len(x)
        # index是mask的坐标
        index = np.empty((length, 2))
        # 这里需要注意的一个问题
        # 提取numpy数组的下标x，y,先行后列
        # 图片中表示坐标，是先宽后高，所以会表示为（y，x）
        index[:, 0] = y
        index[:, 1] = x

        x_center = np.mean(y)
        y_center = np.mean(x)

        # pca降维
        pca = PCA(n_components=1)
        pca.fit(index)
        # X_reduction降维后的数据，一维数组
        X_reduction = pca.transform(index)
        # 对数据进行重构返回，原来的二维数据，但是是线性的
        X_restore = pca.inverse_transform(X_reduction)
        # 对数字进行取整方便画出直线
        X_int = [[round(num) for num in i] for i in X_restore]
        # 这个是画出折线图，如果是线性关系就是直线图
        # 画的数值必须是整数才行
        # 并且需要转为ndarray格式，加上括号
        X_int =np.array(X_int)
        cv2.polylines(mask, [X_int], True, (0, 255, 0), 1)  # pic
        cv2.polylines(image, [X_int], True, (0, 255, 0), 1)  # pic
        # 最小二乘法拟合直线
        #output = cv2.fitLine(index, 2, 0, 0.001, 0.001)

        # print("X_restore length:",len(X_restore))
        first_point = X_restore[0]  # <class 'numpy.ndarray'>
        second_point = X_restore[-1]
        # pointer_k = (y1-y2)/(x1-x2)

        return first_point, second_point, [x_center, y_center]


    def binary_image(self, image):
        condition = image > 0.5
        image[condition] = 1
        image[~condition] = 0
        # image = self.corrosion(image,7)
        image = self.corrosion(image,5)
        # image = self.corrosion(image,3)
        return image

    def corrosion(self, image,kernel_size):
        """
        腐蚀操作
        :param image:
        :return:
        """

        kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image


    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化，将图片缩放到416*416上
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
        return background

if __name__ == '__main__':
    meter_point = meterPoint()
    image = cv2.imread('meter3.jpg')
    start_time = time.time()
    point_vector = meter_point(image)
    end_time = time.time()
    print('inference time:',end_time - start_time)


    '''
    print(type(first_point))
    print(first_point)
    print(second_point)
    cv2.circle(image, (int(first_point[0]), int(first_point[1])), 1, (0, 255, 255), 1)
    cv2.circle(image, (int(second_point[0]), int(second_point[1])), 1, (0, 0, 255), 1)
    cv2.circle(image, (int(center_point[0]), int(center_point[1])), 1, (0, 0, 255), 1)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    '''

