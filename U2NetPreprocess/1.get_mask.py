'''
对json文件进行处理，提取mask生成含有原图和mask的文件夹
'''


import os
path = r'C:\Users\xukechao\Desktop\test'  # path为json文件存放的路径
json_file = os.listdir(path)
# os.system("activate labelme") # activate [自己labelme所在的环境名]
for file in json_file:
    if '.json' in file:
        os.system(r"D:\miniconda3\Scripts\labelme_json_to_dataset.exe %s"%(path + '/' + file))





