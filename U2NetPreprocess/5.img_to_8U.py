import os
import cv2
import numpy as np
path = r'E:\codeWork\yibiao\meter(1)\crop\yibiao_mask\1'
label = []
for file in os.listdir(path):
    if ".png" in file:
        img_path = os.path.join(path,file)
        print(img_path)
        img = cv2.imread(img_path, 0)

        print(np.unique(img))
        #break
        arr = np.uint8(img)

        # for l in np.unique(img):
        #     if l not in label:
        #         label.append(l)
        # print("l: ",label)
        # img[img >1] =1
        #img[img == 38] =1
        #img[img == 76] =1
        # img[img == 75] =2
        #
        #cv2.imwrite(img_path, img)
