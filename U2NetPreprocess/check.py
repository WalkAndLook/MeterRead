import  os

import cv2
import os
import shutil
import random

# path = r'E:\data\diankeyuan\segment\train\mask'
# for file in os.listdir(path):
#     file_name = file.split('_')
#     if len(file_name)==2:
#         file_path = os.path.join(path,file)
#         shutil.move(file_path,r'E:\data\diankeyuan\segment\train\tmp')


path = r'E:\data\diankeyuan\segment\train2'
out_path = r'E:\data\diankeyuan\segment\train2\val'

files = []
for file in os.listdir(path):
    if ".jpg" in file:
        file_path = os.path.join(path,file)
        files.append(file_path)

val_num = int(len(files) * 0.1)
valid = random.sample(files, val_num) #从list中随机获取5个元素，作为一个片断返回

print(valid)

for i in valid:
    mask_path = i.replace(".jpg",".png")

    # file_name = file.replace('.jpg','.png')
    # file_path = os.path.join(a_path,file_name)
    shutil.move(i,out_path)
    shutil.move(mask_path,out_path)


# for file in os.listdir(path):
#     img_path = os.path.join(path,file)
#     mask_path = os.path.join(a_path,file)
#     mask_path2 = mask_path.replace('.jpg','.png')
#
#     img=cv2.imread(img_path)
#     mask_img =cv2.imread(mask_path2)
#
#     if img.shape != mask_img.shape:
#         print(img_path)








