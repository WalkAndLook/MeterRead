import os
import cv2
import numpy as np

from collections import Counter

path = r'E:\codeWork\yibiao\meter(1)\crop\yibiao_mask'
for file in os.listdir(path):
    img_path = os.path.join(path,file)
    print(img_path)
    img = cv2.imread(img_path, 0)
    d = Counter(img.flatten())
    print(np.unique(img))
    # arr = np.uint8(img)

    # img[img == 38] = 1  # point
    # img[img == 75] = 2  # scale

    # img[img == 76] = 1  # point
    # img[img == 75] = 2  # scale
    # cv2.imshow("img",img)
    # cv2.waitKey(0)



