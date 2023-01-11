import os
import json
import cv2
import base64
import numpy as np

def get_boundingbox_point(mask):
    # mask = cv2.imread(mask_path)
    gray_pointer = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts_pointer = cv2.findContours(gray_pointer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_pointer = cnts_pointer[0] if len(cnts_pointer) == 2 else cnts_pointer[1]
    points = []
    pointer_k = []
    for j in range(len(cnts_pointer)):
        try:
            rect = cv2.minAreaRect(cnts_pointer[j])
            box_origin = np.int0(cv2.boxPoints(rect))
            cv2.polylines(mask, [box_origin], True, (0, 0, 255), 1)
            points.append(box_origin.tolist())
        except:
            continue
    # cv2.imshow("mask",mask)
    # cv2.waitKey(0)
    return points

def pointer_scale_maks(mask_path):
    mask = cv2.imread(mask_path)
    mask_poiter = np.copy(mask)
    mask_scale = np.copy(mask)
    mask_poiter[mask == [0, 255, 0]] = 0
    mask_scale[mask == [0, 0, 255]] = 0
    return mask_poiter,mask_scale



def generate_label(points, image_path, h, w,label="scale"):
    N = len(points)
    for i in range(N):
        # 定义一个变量
        boundingbox = {
            "label": "",
            "line_color": None,
            "fill_color": None,
            "points": [],
            "shape_type": "polygon",
            "flags": {}
        }

        boundingbox["label"] = label

        if "erro" in points[i]:
            continue
        else:
            if len(points[i]) >= 4:
                # print(type(points[i]))
                point=points[i]
                point = np.array(point)

                point[:,0]=point[:,0]*(w/416)  #point[0]是宽、point[1]高;image[0]是行是高，image[1]是列是宽
                point[:,1]=point[:,1]*(h/416)

                boundingbox["points"] = point.tolist()

                jsontext["shapes"].append(boundingbox)
                jsondata = json.dumps(jsontext, indent=4, separators=(',', ': '))
                (filename, tempfilename) = os.path.split(image_path)
                filename = filename + "/" + tempfilename.split(".")[-2] + ".json"
                f = open(filename, 'w')
                f.write(jsondata)
                f.close()
            else:
                print("the len of points is less than 3")

def get_json(image, points_pointer, points_scale, image_path):

    global  jsontext
    jsontext= {"version": "3.20.0",
                "shapes": [],
                "lineColor": [],
                "fillColor": [],
                "imagePath": "",
                "imageData": [],
                "imageHeight": 0,
                "imageWidth": 0
                }

    h, w, c = image.shape
    print('width:',w, ' hight:',h)

    imagePath = image_path.split("/")[-1]
    jsontext["imagePath"] = imagePath
    jsontext["imageHeight"] = h
    jsontext["imageWidth"] = w

    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        s = base64_data.decode()
        jsontext["imageData"] = s
    generate_label(points_pointer, image_path, h, w ,label="pointer")
    generate_label(points_scale, image_path, h, w,label="scale")

def image_resize(img):
    h, w, c = img.shape

    if w > h:
        img = cv2.copyMakeBorder(img, (w - h) // 2, (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w < h:
        img = cv2.copyMakeBorder(img, 0, 0, (h - w) // 2, (h - w) // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        pass
    img2 = cv2.resize(img, (416, 416))
    return img2


# mask_path = r'E:/data/2/0001_.png'
# img_path = r'E:/data/2/0001.jpg'

for root,dir,files in os.walk(r"E:\data\diankeyuan\segment\test\youwen"):
    for file in files:
        if ".jpg" in file:
            img_path = os.path.join(root,file)
            mask_path = img_path.replace(".jpg","__.png")
            # print(img_path)
            # print(mask_path)
            print("img_path: ",img_path)
            image = cv2.imread(img_path)
            image = image_resize(image)

            # image = cv2.imread(img_path)
            # image = image_resize(image)
            # cv2.imwrite(img_path,image)

            mask_poiter,mask_scale = pointer_scale_maks(mask_path)
            poiter_points = get_boundingbox_point(mask_poiter)
            print("poiter_points",poiter_points)
            scale_points = get_boundingbox_point(mask_scale)
            get_json(image, poiter_points, scale_points, img_path)
