import cv2
import numpy as np
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
from proto import detection_pb2_grpc as pd2_grpc
from utility.logger import Logger
from utility import functions as func
from .yolov5.inference import DetectionYolov5


logger = Logger().logger
# 提供对外接入服务
class ObjectGrpcService(pd2_grpc.ioper_serverServicer):
    def __init__(self):
        self.data = func.load_config()['station_dlstate']
        self.object_detection_yolov5 = DetectionYolov5()

    def ObjectDetection(self, request, context):
        results = []

        try:
            thresh = request.thresh
            iou_thresh = request.iou_thresh
            current_image = request.current_image
            # 对输入的图片string做处理，转为OpenCV的BGR格式
            image = func.base64_to_mat(current_image)
        except:
            return pd2.ObjectDetectionResponse(result=-1, object_bbox=[], result_status=func.result_info(1))
        logger.info("GetObjectDetectionFromImage request =========================================")
        logger.info("input image shape       :{}".format(image.shape))
        logger.info("yolov5 confidence thresh:{}".format(thresh))
        logger.info("yolov5 iou_thresh       :{}".format(iou_thresh))

        if image is None:
            return pd2.ObjectDetectionResponse(result=-1, object_bbox=[], result_status=func.result_info(1))

        if thresh is None or thresh <= 0:
            thresh = self.data['Pipeline']['detection']['threshold']

        if iou_thresh is None or iou_thresh <= 0:
            iou_thresh = self.data['Pipeline']['detection']['iou_threshold']

        if self.data['Pipeline']['isopen']:
            bbox_detection = self.object_detection_yolov5.detect(image, thresh, iou_thresh)

            for ix, one_result in enumerate(bbox_detection):
                label, xmin, ymin, xmax, ymax, confidence = one_result[0:6]
                if confidence >= thresh:
                    results.append([label, confidence, xmin, ymin, xmax, ymax])

            if len(results) == 0:
                return pd2.ObjectDetectionResponse(result=0, object_bbox=[], result_status=func.result_info(2))

            response_result = []

            try:
                for one_result in results:
                    response_result.append(
                        pd2.DetectionBbox(
                            label=one_result[0],
                            confidence=one_result[1],
                            xmin=one_result[2],
                            ymin=one_result[3],
                            xmax=one_result[4],
                            ymax=one_result[5]
                        )
                    )
                return pd2.ObjectDetectionResponse(result=0, object_bbox=response_result, result_status=func.result_info(0))
            except Exception as e:
                return pd2.ObjectDetectionResponse(result=-1)


