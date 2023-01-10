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

img_path = r'yibiao2.jpg'
def run():
    with open(img_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read())

    conn = grpc.insecure_channel('127.0.0.1:5000')
    client = pd2_grpc.ioper_serverStub(channel=conn)
    request = pd2.ReadMeterRequest(
        thresh=0.3,
        iou_thresh=0.5,
        current_image=img_base64
    )
    start = time.time()
    response = client.ReadMeter(request)
    end = time.time()
    print("time:", end - start)
    print("response.result:", response.result)
    print("response point vector:\n", response.vector_list)


if __name__ == '__main__':
    run()
