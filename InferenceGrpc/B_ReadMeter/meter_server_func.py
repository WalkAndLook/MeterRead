'''
该程序是用于在ioper_server里调用，
带1的程序是测试代码用的
'''
import cv2
import numpy as np
import base64
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT1 = FILE.parents[1]
if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))

from proto import detection_pb2 as pd2
# from proto import detection_pb2_grpc as pd2_grpc
from utility.logger import Logger
from utility import functions as func
from B_ReadMeter.yolov5.inference import DetectionYolov5
from B_ReadMeter.U2Net.inference import meterPoint


logger = Logger().logger
# 提供对外接入服务
class MeterGrpcService():
    def __init__(self):
        self.data = func.load_config()['ReadMeter']
        self.object_detection_yolov5 = DetectionYolov5()
        self.meter_point = meterPoint()

    def ReadMeter(self, request, context):
        try:
            thresh = request.thresh
            iou_thresh = request.iou_thresh
            current_image = request.current_image
            # 对输入的图片string做处理，转为OpenCV的BGR格式
            image = func.base64_to_mat(current_image)
        except:
            return -1
        logger.info("meter image detection with yolov5  =========================================")
        logger.info("input image shape        :{}".format(image.shape))
        logger.info("yolov5 confidence thresh :{}".format(thresh))
        logger.info("yolov5 iou_thresh        :{}".format(iou_thresh))

        if image is None:
            logger.error('Exception: image is None !')
            return pd2.ReadMeterResponse(result=-1, vector_list=[])

        if thresh is None or thresh <= 0:
            thresh = self.data['detect_meter']['threshold']

        if iou_thresh is None or iou_thresh <= 0:
            iou_thresh = self.data['detect_meter']['iou_threshold']

        bbox_detection = self.object_detection_yolov5.detect(image, thresh, iou_thresh)

        point_result = []
        for ix, one_result in enumerate(bbox_detection):
            label, xmin, ymin, xmax, ymax, confidence = one_result[0:6]
            if label == "yibiao" and confidence > thresh:
                logger.info('start meter pointer segment===========================')
                box = one_result[1:5]
                img_meter = func.clip_image(image, box)
                point_vector = self.meter_point(img_meter)   # 返回指针向量[x, y]
                point_result.append(point_vector)
            else:
                pass
        if len(point_result) == 0:
            logger.info('detect no point or meter!')
            return pd2.ReadMeterResponse(result=0, vector_list=[])
        else:
            vector_list = []
            for vector in point_result:
                vector_list.append(pd2.pointVector(
                    x=vector[0],
                    y=vector[1]
                ))
            logger.info('return meter point vector in list [x, y]===============')
            return pd2.ReadMeterResponse(result=0, vector_list=vector_list)


if __name__ == '__main__':
    pass
    



