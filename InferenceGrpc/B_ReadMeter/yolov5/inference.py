import os
import platform
import sys
from pathlib import Path
import json
import torch
import base64
import numpy as np
# 添加导包路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT1 = FILE.parents[2]
if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))

from models.common import DetectMultiBackend
from utils.general import (Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           letterbox)
from utility.logger import Logger
import utility.functions as func

logger = Logger().logger

class DetectionYolov5(object):
    def __init__(self):
        self.data = func.load_config()['station_dlstate']['Pipeline']['detection']
        self.device = self.get_device(self.data['gpu_id']) if self.data['gpu'] else 'cpu'
        self.device = torch.device(self.device)
        imgsz = self.data["img_size"]
        path = Path(__file__).parent  # 这里用于表示父目录到yolov5l目录下
        yaml_path = '{}/models/yolov5l.yaml'.format(path)
        data_path = '{}/images/mydata.yaml'.format(path)
        weights = '{}/weights/best.pt'.format(path)
        # 加载模型
        self.model = DetectMultiBackend(weights, device=self.device, data=data_path)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image siz

    def detect(self, image, conf_thres, iou_thres):
        im, im0s = self.ReadImages(image, img_size=self.imgsz, stride=self.stride)
        # run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        logger.info(f'Now run model inference with thresh:{conf_thres}, iou_thresh:{iou_thres}')
        with dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        with dt[1]:
            # 执行模型推理
            pred = self.model(im, augment=False)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None)
        # write result box
        bbox = []
        logger.info('return the inference result in list: label, xmin, ymin, xmax, ymax, conf')
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] =scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xmin = int(xyxy[0].item())
                    ymin = int(xyxy[1].item())
                    xmax = int(xyxy[2].item())
                    ymax = int(xyxy[3].item())
                    conf = conf.item()
                    cls = cls.item()
                    c = int(cls)
                    if conf >= conf_thres:
                        label = self.names[c]
                        bbox.append([label, xmin, ymin, xmax, ymax, conf])

        return bbox

    def ReadImages(self, image, img_size=640, stride=32):
        im0 = image
        im = letterbox(im0, img_size, stride=stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return im, im0


    @staticmethod
    def get_device(id):
        return "cuda:%s" % (str(id)) if torch.cuda.is_available() else "cpu"

    def destroy_model(self):
        torch.cuda.empty_cache()
        self.model = None

def base64_to_mat(data):
       # base64解码，传入的图片是字符串，需要解码成图片，本地的时候是直接输入地址获取图片，传输的时候是通过字符串传输
       strDecode = base64.b64decode(data)
       # 转换np序列
       byteDecode = np.frombuffer(strDecode, np.uint8)
       # 转换OpenCV格式
       return cv2.imdecode(byteDecode, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    meterYolo = DetectionYolov5()
    img_path = 'image/yibiao.jpg'

    with open(img_path, 'rb') as f:
        obj_image_base64 = base64.b64encode(f.read())
    image = base64_to_mat(obj_image_base64)

    thresh = 0.1
    iou_thresh = 0.45

    result = meterYolo.detect(image, thresh, iou_thresh)

    print(result)
