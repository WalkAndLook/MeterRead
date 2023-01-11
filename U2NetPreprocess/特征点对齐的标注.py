import os
import json
import base64
import cv2
import numpy as np

import urllib3

http = urllib3.PoolManager()

# r'E:\data\yibiao2\bileiqi_0-3\0001.jpg',
# r'E:\data\yibiao2\SF6/0001.jpg',
# r'E:\data\yibiao2\youowen_-20_140\0001.jpg',
# r'E:\data\yibiao2\youweiji_0_1_90\0001.jpg'

path = r'H:\NanJingSongJian\train\yi1\2'
base_json_file = r'{}\20220806_000760.json'.format(path)


img_paths = [os.path.join(path, x) for x in os.listdir(path)]

def dst_points(points,M):
    points = np.array(points)
    dst_x = (M[0][0]*points[:,0] + M[0][1]*points[:,1] + M[0][2])/(M[2][0]*points[:,0] + M[2][1]*points[:,1] + M[2][2])
    dst_y = (M[1][0]*points[:,0] + M[1][1]*points[:,1] + M[1][2])/(M[2][0]*points[:,0] + M[2][1]*points[:,1] + M[2][2])
    dst_x = np.expand_dims(dst_x,axis = 1)
    dst_y = np.expand_dims(dst_y,axis = 1)
    return np.concatenate((dst_x, dst_y), axis=1).tolist()


for img_path in img_paths:
    filepath, fullflname = os.path.split(img_path)
    fname, ext = os.path.splitext(fullflname)
    if ext in ['.png','.jpg']:
        with open(base_json_file, "r", encoding='utf-8') as jsonFile:
            json_data = json.load(jsonFile)
            base_img_base64 = json_data['imageData']

            with open(img_path, 'rb') as f:
                base64_data2 = base64.b64encode(f.read())
                align_img_base64 = base64_data2.decode()

            data = {"obj_image_data": align_img_base64,
                    "current_image_data": base_img_base64,
                    "is_have_image_obj": False,
                    "image_obj_xmin": 0,
                    "image_obj_ymin": 0,
                    "image_obj_xmax": 0,
                    "image_obj_ymax": 0
                    }
            encoded_data = json.dumps(data).encode('utf-8')
            # 发送一个body的json数据
            r = http.request('POST', 'http://10.16.196.42:10092/superglue', body=encoded_data,
                             headers={'Content-Type': 'application/json'})
            M = np.array(json.loads(r.data.decode('utf-8'))['h'])

            for i in range(len(json_data["shapes"])):
                points = json_data["shapes"][i]["points"]
                dst_points_x_y = dst_points(points,M)
                json_data["shapes"][i]["points"] = dst_points_x_y

            json_data["imagePath"] = os.path.basename(img_path)
            json_data['imageData'] = align_img_base64

            new_json_path = img_path.replace('.jpg', '.json')
            print("joon_path:",new_json_path)
            with open(new_json_path, "w") as json_path:
                json.dump(json_data, json_path, indent=4, separators=(',', ': '))
