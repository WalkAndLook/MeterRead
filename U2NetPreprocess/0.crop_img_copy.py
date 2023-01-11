'''
代码功能实现：
把原图片和标签文件复制粘贴到了crop文件里
'''

# 导入模块
import shutil

import cv2
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import numpy as np
import random

# 原图片、标签文件、裁剪图片路径
img_path = r'images'
obj_img_path = r'crop'
xml_path = img_path

# 声明一个空字典用于储存裁剪图片的类别及其数量
Numpic = {}

count = 0
# 把原图片裁剪后，按类别新建文件夹保存，并在该类别下按顺序编号
for img_file in os.listdir(img_path):
    try:
        if img_file[-4:] in ['.png', '.jpg']:  # 判断文件是否为图片格式
            img_filename = os.path.join(img_path, img_file)  # 将图片路径与图片名进行拼接
            print("img_filename:",img_filename)
            xml_path_ = img_filename.replace(".jpg",".xml")
            print("xml_path:",xml_path_)
            # img_cv = cv2.imread(img_filename)  # 读取图片
            image = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
            img_name = (os.path.splitext(img_file)[0])  # 分割出图片名，如“000.png” 图片名为“000”
            xml_name = xml_path + '\\' + '%s.xml' % img_name  # 利用标签路径、图片名、xml后缀拼接出完整的标签路径名
            print("xml_name:",xml_name)


            if os.path.exists(xml_name):  # 判断与图片同名的标签是否存在，因为图片不一定每张都打标
                print(xml_name)
                root = ET.parse(xml_name).getroot()  # 利用ET读取xml文件
                for obj in root.iter('object'):  # 遍历所有目标框
                    name = obj.find('name').text  # 获取目标框名称，即label名
                    xmlbox = obj.find('bndbox')  # 找到框目标
                    x0 = xmlbox.find('xmin').text  # 将框目标的四个顶点坐标取出
                    y0 = xmlbox.find('ymin').text
                    x1 = xmlbox.find('xmax').text
                    y1 = xmlbox.find('ymax').text

                    # obj_img = img_cv[int(float(y0)):int(float(y1)), int(float(x0)):int(float(x1))]  # cv2裁剪出目标框中的图片

                    Numpic.setdefault(name, 0)  # 判断字典中有无当前name对应的类别，无则新建
                    Numpic[name] += 1  # 当前类别对应数量 + 1
                    my_file = Path(obj_img_path + '\\' + name)  # 判断当前name对应的类别有无文件夹
                    if 1 - my_file.is_dir():  # 无则新建
                        os.mkdir(obj_img_path + '\\' + str(name))

                    try:
                        print("xml_path_",xml_path_)
                        shutil.copy(xml_path_,obj_img_path+"\\"+name)
                    except Exception as e:
                        pass


                    shutil.copy(img_filename, obj_img_path + "\\" + name)
                    # cv2.imwrite(obj_img_path + "\\"+name +"\\"+img_file,obj_img)  # 保存裁剪图片，图片命名4位，不足补0
                    # cv2.imwrite(obj_img_path + '\\' + name + '\\' + '%04d' % (count) + '_base.jpg', obj_img)  # 保存裁剪图片，图片命名4位，不足补0
                    count+=1
                    break
    except Exception as e:
        continue