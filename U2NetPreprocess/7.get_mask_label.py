import os
import random

#修改源码之后 这一步可以省略--change by chlg 2021.11.24

img_train_path = r'E:\prj\ReadMeter-master\data\images'
train_path = r'E:\prj\ReadMeter-master\data\train.txt'
val_path = r'E:\prj\ReadMeter-master\data\val.txt'

all_images = []

for file in os.listdir(img_train_path):
    all_images.append(file)

val_num = int(len(all_images) * 0.1)

valid = random.sample(all_images, val_num)
train = set(all_images) - set(valid)

with open(val_path, 'w', encoding='utf-8') as fw:
    for line in valid:
        txt = 'images/val/' + line + ' ' + 'annotations/val/' + line.replace('.jpg', '.png')
        fw.write(txt)
        fw.write('\n')
with open(train_path, 'w', encoding='utf-8') as fw2:
    for line2 in train:
        txt2 = 'images/train/' + line2 + ' ' + 'annotations/train/' + line2.replace('.jpg', '.png')
        fw2.write(txt2)
        fw2.write('\n')






