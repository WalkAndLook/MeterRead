# Folder Target
This folder is U2Net model. We use the model to trian the pointer of meter segmentation, and test the result of segmentation.In this folder, we test the model and return
the segmentation image with test_meter.py. But when we inference with the model, we will just return the pointer vector of meter in folder InferenceGrpc.
# Folder Content
+ data: save the meter data for train and test，the data has been processed in folder U2NetPreprocess, it saved format just like meter image in jpg and mask image
in png
+ degree_calculate： this folder is used to calculate the degree of meter when the pointer is segmented and return the pointer vector in inference file. When we
input the meter other parameters such as the position of center, min and max, it will return the degree. Notice: the inference file is in InferenceGrpc, not in this
folder U2Net.
+ font: save one ttf font used in code.
+ models: net.py saved the U2Net model code.
+ other_code: some other type segment train and test python file
+ result/meter4: saved the test file segment result image.
+ utiles: save the dataset python file and the model loss function file
+ video: we don't use it if what we input is image
+ weight: save the model weight. the initial model weight will be given next part in BaiduNetdisk, the trained model is also in BaiduNetdisk.
+ train_meter.py : the train python file
+ test_meter.py : the test python file, it return the segment result image, not the pointer vector in inference
# Using Method
1. train_meter.py : train the model
2. test_meter.py : test the model weight results  
Notice: the specific method is in <https://note.youdao.com/s/ZLgzyfvU>
# Weight Download
the initial u2net weight net.pt  
url: <https://pan.baidu.com/s/1f2cC3lNdrJTCWwoBul0Oug?pwd=3wz0>  
password：3wz0  
the one trained u2net weight net_meter.pt  
see it in folder InferenceGrpc
