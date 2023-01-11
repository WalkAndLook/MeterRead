'''
将步骤1中的生成的含有原图与mask图的_json文件夹里面的图片和mask图片提取出来
放到新的路径文件下，jpg为原图，png为mask，此时mask显示为红色，绿色可以看见的颜色2
'''

import os
import shutil

import cv2


path = r'crop/yibiao'  # 输入路径，包含原图和mask的_json文件夹
out_path = r'crop/yibiao_mask' # 输出路径，输出原图jpg和mask图png的文件夹，此时的label文件颜色为红色或者绿色，是可以看到的颜色

if not os.path.exists(out_path):
    os.mkdir(out_path)

for dir in os.listdir(path):
    if '_json' in dir:
        for file in os.listdir(os.path.join(path,dir)):
            if file =='label.png':
                mask_path = os.path.join(path,dir,file)
                img_path = os.path.join(path,dir,file.replace('label.png','img.png'))

                mask = cv2.imread(mask_path)
                image =  cv2.imread(img_path)

                new_file_name = dir.replace('_json','')
                new_mask_path = os.path.join(out_path,new_file_name+'.png')
                new_img_path = os.path.join(out_path,new_file_name+'.jpg')

                cv2.imwrite(new_mask_path,mask)
                cv2.imwrite(new_img_path,image)

                print(new_mask_path)



# for root,dir,files in os.walk(path):
#     # print(dir)
#
#     for file in files:
#         if file =='label.png':
#             img_path = os.path.join(root,file)
#             # new_img_path
#
#             print(img_path)