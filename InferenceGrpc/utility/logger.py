# 日志书写

import time
import logging
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
path = FILE.parents[1]
# print(path) # E:\codeWork\pythonWork\ioper_kc
class Logger(object):
    logger = None
    def __init__(self):
        if Logger.logger is None:
            Logger.logger = logging.getLogger()  # 创建一个logger
            Logger.logger.setLevel(logging.DEBUG)
            self.log_time = time.strftime("%Y_%m_%d_")
            self.log_path = str(path) + "/log/"

            if not os.path.exists(self.log_path):
                os.mkdir(self.log_path)

            self.log_name = self.log_path + self.log_time + 'log.log'

            fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 文件输出
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            # [2022-12-07 10:13:08,390] [INFO] load model SViT_E1_D1_16.pth finished [inference.py:53]
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s [%(filename)s:%(lineno)d]'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            Logger.logger.addHandler(fh)  # 给logger添加handler
            Logger.logger.addHandler(ch)

            fh.close()
            ch.close()

if __name__ == '__main__':
    logger = Logger().logger
    logger.info("this is a fan")
