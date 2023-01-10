import json
import torch
import base64
import math
import numpy as np
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
path = FILE.parents[0]

def load_config():
    return json.load(open(str(path) + "\config.json", "r", encoding='utf-8'))

# con = load_config()
# print(con)

def get_device(id):
    return "cuda:%s" % (str(id)) if torch.cuda.is_available() else "cpu"

def result_info(index, s1="", s2="", s3=[]):
    info = {
        0: "",
        1: f"input {s1}image{s2} is empty!",
        2: "no results detected!",
        3:  f" Format of input Group is error! Your input Group is {s3}. The correct example format is [] or ['0001', '0121', '0006', ...]. "
    }
    return info[index]


def base64_to_mat(data):
    # base64解码，传入的图片是字符串，需要解码成图片，本地的时候是直接输入地址获取图片，传输的时候是通过字符串传输
    strDecode = base64.b64decode(data)
    # 转换np序列
    byteDecode = np.frombuffer(strDecode, np.uint8)
    # 转换OpenCV格式
    return cv2.imdecode(byteDecode, cv2.COLOR_BGR2RGB)

# 裁剪图片
def clip_image(img, box):
    '''
    :param img: 输入的图片
    :param box: 裁剪的位置，box是list，box=[xmin,ymin,xmax,ymax]
    :return: 裁剪后的图片
    '''
    box = np.array(box)
    box = np.where(box < 0, 0, box)  # 超出图片边界的位置置0
    return img[box[1]:box[3], box[0]:box[2]]

# 计算两点直接的距离
def point_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ^ 2 + (y1 - y2) ^ 2)
