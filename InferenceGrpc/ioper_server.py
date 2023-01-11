import time
import grpc
from concurrent import futures
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from proto import detection_pb2 as pd2
from proto import detection_pb2_grpc as pd2_grpc
from utility.logger import Logger
from utility import functions as funcs
from A_ObjectDetect.object_server_func import ObjectGrpcService
from B_ReadMeter.meter_server_func import MeterGrpcService

logger = Logger().logger
config = funcs.load_config()


class GrpcServer(pd2_grpc.ioper_serverServicer):
    def __init__(self):
        if config['station_dlstate']['open']:
            objectService = ObjectGrpcService()
            self.ObjectDetection = objectService.ObjectDetection

        if config['ReadMeter']['open']:
            meterService = MeterGrpcService()
            self.ReadMeter = meterService.ReadMeter

    def buid_grpc_server(self, ip='0.0.0.0', port='5000', workers=4, send_msg_len=100*1024*1024, receive_msg_len=100*1024*1024):
        # 启动grpc服务
        grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=[
                ('grpc.max_send_message_length', send_msg_len),
                ('grpc.max_receive_message_length', receive_msg_len)
            ]
        )
        # 注册服务到grpc里
        pd2_grpc.add_ioper_serverServicer_to_server(self, grpc_server)
        grpc_server.add_insecure_port(ip + ':' + port)
        logger.info(f"build grpc server,ip:{ip},port:{port},workers:{workers}")
        return grpc_server

def run():
    grpc_server = GrpcServer()
    grpc_server_start = grpc_server.buid_grpc_server(ip='0.0.0.0', port='5000')
    grpc_server_start.start()
    logger.info("================================objection detect server has been started============================")
    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        grpc_server_start.stop(0)

if __name__ == '__main__':
    run()





