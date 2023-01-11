# coding:utf-8
# 客户端

import grpc
import base64
import time
import cv2
import numpy as np
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from proto import detection_pb2 as pd2
from proto import detection_pb2_grpc as pd2_grpc

img_path = r'yibiao.jpg'

def draw_bbox(bbox, img0, color, wt):
    det_result_str = ''
    for idx in range(len(bbox)):
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
    return img0

def run():
    with open(img_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read())

    conn = grpc.insecure_channel('127.0.0.1:5000')
    client = pd2_grpc.ioper_serverStub(channel=conn)
    request = pd2.ObjectDetectionRequest(
        thresh=0.3,
        iou_thresh=0.5,
        current_image=img_base64
    )
    start = time.time()
    response = client.ObjectDetection(request)
    end = time.time()
    print("time:", end - start)
    print("response.result:", response.result)
    print("response.bbox:", response.object_bbox)
    print("response.result_status:", response.result_status)

    bbox = [[x.xmin, x.ymin, x.xmax, x.ymax] for x in response.object_bbox]
    print('bbox:', bbox)

    img0 = cv2.imread(img_path)
    img0 = draw_bbox(bbox, img0, (0, 0, 255), 5)
    cv2.imwrite('yibiao_res.jpg', img0)
    print("the result image has been saved!!!")

if __name__ == '__main__':
    run()
