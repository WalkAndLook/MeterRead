import os
import random
import shutil
import numpy as np

import cv2

# path  = r'E:\data\diankeyuan\segment\train2\train\0006_base_1.png'

def random_crop(mask):

    h,w  = mask.shape[0:2]


    print("mask.shape", mask.shape)
    not_none = np.argwhere(mask>0)
    print("not_none.shape",not_none.shape)

    min_x = np.min(not_none[:,1])
    min_y = np.min(not_none[:,0])
    max_x = np.max(not_none[:,1])
    max_y = np.max(not_none[:,0])

    # print(min_x,min_y,max_x,max_y)

    min_x = int(w*0.1)
    min_y = int(h*0.1)
    max_x = int(w*0.9)
    max_y = int(h*0.9)
    return  min_x,min_y,max_x,max_y

path = r'E:\data\zhangbei\crop\yibiao\mask'
for file in os.listdir(path):
    if ".png" in file:
        mask_path = os.path.join(path,file)
        img_path = mask_path.replace(".png",".jpg")

        mask = cv2.imread(mask_path, 0)
        print(mask.shape)
        h,w = mask.shape[0:2]
        img = cv2.imread(img_path,1)
        min_x, min_y, max_x, max_y = random_crop(mask)

        print("min_x: ",min_x)
        print("min_y: ",min_y)
        print("max_x: ",max_x)
        print("max_y: ",max_y)

        min_x = random.randint(0,min_x)
        min_y = random.randint(0,min_y)
        max_x = random.randint(max_x, w)
        max_y = random.randint(max_y, h)
        img = img[min_y:max_y,min_x:max_x]
        mask = mask[min_y:max_y,min_x:max_x]

        mask_path_out =  mask_path.replace(".png","_crop.png")
        img_path_out = img_path.replace(".jpg","_crop.jpg")

        cv2.imwrite(mask_path_out, mask)
        cv2.imwrite(img_path_out, img)




