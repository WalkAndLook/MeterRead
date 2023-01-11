# Folder Target
This project is build for master paper. It mainly contained two parts, the pointer meter and the digital meter automatically degree read. Now the folder only have pointer
meter degree read method, I will put different part work in it.
# Folder Content
+ InferenceGrpc: this folder is used for inference the model, it is build with grpc, so that it can be remote call by others and the trained well model can be used 
+ U2Net : this is U2Net model for pointer segment
+ U2NetPreprocess : this folder is used for mask image preprocessing when image is labeled by labelme, so that the image and mask data can be used by u2net model
+ Yolov5 : this is Yolov5 model. It will be used for meter detection from image, then the detection meter will be send to U2Net model for pointer segmentation.
# Using Method
+ Step1: using the Yolov5 train the model for meter detection and test the trained model to see the results is good or not
+ Step2: After label the detection meter image by labelme, I will process the data in U2NetPreprocess so that the data format fit segmentation model u2net. Then U2Net
model will train the data and test the weight result. When the Yolov5 and U2net weight is ok for test image, I will use them inference in folder InferenceGrpc.
+ Step3: use the Step1 and Step2 trained weight, see inference result is ok or not in InferenceGrpc. It will return the pointer vector of meter, so it can be used to get
the meter degree in file U2Net/degree_calculate/test_calculate.py
