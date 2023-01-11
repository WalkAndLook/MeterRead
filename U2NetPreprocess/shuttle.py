import os
import shutil
import random


path = r'H:\wenzishibie\det\icdar_c4_train_imgs'
out_path  = r'H:\wenzishibie\det\ch4_test_images'

files_path = []
for root, dir,files in os.walk(path):
    for file in files:
        if '.jpg' in file:
            file_path = os.path.join(root,file)
            print(file_path)
            files_path.append(file_path)


val_num = int(len(files_path)*0.1)
valid = random.sample(files_path, val_num)
for f in valid:
    print("f: ",f)
    xml_path = f.replace(".jpg",".json")
    shutil.move(f,out_path)
    shutil.move(xml_path,out_path)