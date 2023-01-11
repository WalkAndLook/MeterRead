'''
将图片resize到416*416
目的是后面做指针旋转，图片增广
'''

import os

import cv2
import numpy as np

def img_resize(image, image_size):
    # 参数---image：图片，image_size：缩放后的图片大小
    # 先按照一定比例去缩放
    # 再放到0416*416的背景图里
    # 结果图片会在宽或高的地方出现一些黑边
    h1, w1, _ = image.shape
    max_len = max(h1, w1)
    fx = image_size / max_len
    fy = image_size / max_len
    # 缩小图像时，使用INTER_AREA插值方式效果最好
    # 区域插值法。就是根据当前像素点周边区域的像素实现当前像素点的采样。
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

    h2, w2, _ = image.shape
    background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # background[:, :, :] = 127
    background[:, :, :] = 0
    s_h = image_size // 2 - h2 // 2
    s_w = image_size // 2 - w2 // 2
    background[s_h:s_h + h2, s_w:s_w + w2] = image
    image = background.copy()
    # print("image.shape:", image.shape)
    return image

path =r'crop/yibiao_mask'
for file in os.listdir(path):
    file_path = os.path.join(path,file)
    image = cv2.imread(file_path)
    image2 = img_resize(image,416)
    cv2.imwrite(file_path,image2)
    print(file_path)


