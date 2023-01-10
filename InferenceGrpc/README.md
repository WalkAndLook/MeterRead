# Folder Target 
This project is used for inference. When we trained a model with data and test results is good, we hope others or other project can use the model with the wights. So the inference is needed.
The folder is writed for this. By using grpc protocol, we can use the model with remote call, so that the project can be used in industry scene.
# Folder Content
+ A_ObjectDetect  : model inference for object detection with yolov5  
  + yolov5       : yolov5 model and inference file    
  + object_server_func.py   : ioper_server.py will use it, this file is a function, receive the parameters from the client and send it to yolov5 model inference file and 
  return we need result data form.  
+ B_ReadMeter    : model inference for pointer_meter automatic degree reading with yolov5+u2net  
  + U2Net   : model for segmenting the pointer of meter ,and return the pointer vector from meter center to needle tip  
  + yolov5 : model for detecting the meter position, and clip the meter image to segment the pointer with U2Net  
  + meter_server_func.py : ioper_server.py will use it, this file is a function like object_server_func.py
+ log            : save logs file in the project    
+ proto          : grpc request and response parameter  
+ test_client    : the clien for testing the project mdoel in A_ObjectDetect and B_ReadMeter  
+ utility        : saved some small tools,such as commen functions and logger  
  + config.json : the json file saved some model parameters, such as thresh and iou_thresh for yolov5, it will be used in model parameter input  
  + functions.py : this file saved some commen functions that maybe used in other python file, such as the model inference.py
  + logger.py : this file is log file, it write the log class ,so that the whole project can output the middle result
+ ioper_server.py: the server for the whole project in grpc  

# Using Method
+ first: run the ioper_server.py ,so that the grpc service will be started!!!  And please keep the python file don't stop, keep the service going on until the client will
not used.
+ second: run the different client.py under test_client folder, input the new parameters, it will return model inference results

# Weight Download
+ yolov5 weights
  + position: A_ObjectDetect/yolov5/weights and B_ReadMeter/yolov5/weights
  + url: <https://pan.baidu.com/s/1P8McW3KGL4656p7xCpYe9w?pwd=ho5u>
  + password: ho5u
+ U2Net weights
  + position: B_ReadMeter/U2Net/weight
  + url: <https://pan.baidu.com/s/14K2zscZw6az3KtJu7G6EgQ?pwd=lb7h>
  + password: lb7h

