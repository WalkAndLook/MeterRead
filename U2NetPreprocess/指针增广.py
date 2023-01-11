import cv2
import random
import numpy as np

class Aug_image(object):
    def __init__(self,image):
        self.image = image

    def img_resize(self,image, image_size):
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        image = background.copy()
        print("image.shape:", image.shape)
        return image

    def dumpRotateImage(self, img, degree,pointer):
            height, width = img.shape[:2]
            M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
            new_pointer = [int(M[0][0] * pointer[0] + M[0][1] * pointer[1] + M[0][2]),
                       int(M[1][0] * pointer[0] + M[1][1] * pointer[1] + M[1][2])]

            # new_pointer = np.dot(M, np.array([[pointer[0]], [pointer[1]], [1]]))
            # print("new:",new_pointer)
            return new_pointer


path = r'E:\data\0\0001_.png'
img = cv2.imread(path,1)
kernel = np.ones((7,7),np.uint8)
img = cv2.dilate(img,kernel)

img2 = img.copy()


aug = Aug_image(img)

img3 = cv2.imread(r'E:\data\0\0001.jpg',1)
img3 = aug.img_resize(img3,416)
img4 = img3.copy()

# img4 = cv2.imread(r'E:\data\0\0001.jpg',1)
# img4 = aug.img_resize(img4,416)

# img4 = cv2.addWeighted(img2, 0.6, img3, 0.4, 0)
# cv2.imshow('img3',img3)
# cv2.waitKey(0)
b, g, r = cv2.split(img)
pointer_index = np.where(r==255)

pointers = zip(pointer_index[0],pointer_index[1])
new_pointers = []
degree = random.randint(30, 80) #旋转角度

mask2 = np.zeros(r.shape).astype('uint8')
for i,v in pointers:
    pointer = [i,v]
    pointer_new = aug.dumpRotateImage(r, degree,pointer)
    new_pointers.append(pointer_new)
    img3[i,v] = img3[pointer_new[0],pointer_new[1]]
    mask2[pointer_new[0], pointer_new[1]] = 255

pointer_new_index = np.where(mask2==255)
pointers_new = zip(pointer_new_index[0],pointer_new_index[1])

print("pointer_new_index",pointer_new_index)
h,w  = mask2.shape
for i,v in pointers_new:
    mask2[i+1,v+1] = 255
    mask2[i-1,v-1] = 255
    mask2[i+1,v-1] = 255
    mask2[i-1,v+1] = 255

cv2.imshow('mask2', mask2)
cv2.waitKey(0)

pointer_new_new_index = np.where(mask2==255)
pointers_new_new = zip(pointer_new_new_index[0],pointer_new_new_index[1])

for i,v in  pointers_new_new:
    pointer = [i,v]
    pointer_new = aug.dumpRotateImage(mask2, -degree,pointer)
    # new_pointers.append(pointer_new)
    img3[i,v] = img4[pointer_new[0],pointer_new[1]]
    # mask2[pointer_new[0], pointer_new[1]] = 255


cv2.imshow('img3',img3)
cv2.imwrite('img3.png',img3)
cv2.waitKey(0)


# import cv2
# import random
# import numpy as np
#
# def dumpRotateImage(img, degree,pointer):
#     height, width = img.shape[:2]
#     M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
#     new_pointer = [int(M[0][0] * pointer[0] + M[0][1] * pointer[1] + M[0][2]),
#                int(M[1][0] * pointer[0] + M[1][1] * pointer[1] + M[1][2])]
#
#     # new_pointer = np.dot(M, np.array([[pointer[0]], [pointer[1]], [1]]))
#     # print("new:",new_pointer)
#
#     return new_pointer
#
#
# def contour(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(img, contours, -1 ,(0,0,255),3)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray[gray[:,:]!=0]=255
#
#     return gray
#
#
#
# def ime_resize(image,image_size):
#     h1, w1, _ = image.shape
#     max_len = max(h1, w1)
#     fx = image_size / max_len
#     fy = image_size / max_len
#     image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
#     h2, w2, _ = image.shape
#     background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
#     background[:, :, :] = 127
#     s_h = image_size // 2 - h2 // 2
#     s_w = image_size // 2 - w2 // 2
#     background[s_h:s_h + h2, s_w:s_w + w2] = image
#     image = background.copy()
#     print("image.shape:",image.shape)
#     return image
#
#
#
# path = r'E:\data\0\0001_.png'
# img = cv2.imread(path,1)
# kernel = np.ones((5,5),np.uint8)
#
# img = cv2.dilate(img,kernel)
# img2 = cv2.imread(r'E:\data\0\0001.jpg',1)
# img2 = ime_resize(img2,416)
#
# # cv2.imshow("img2",img2)
# img3 = img2.copy()
#
# b, g, r = cv2.split(img)
# pointer_index = np.where(r==255)
#
# pointers = zip(pointer_index[0],pointer_index[1])
# new_pointers = []
# degree = random.randint(20, 30) #旋转角度
#
#
# mask2 = np.zeros(r.shape).astype('uint8')
# for i,v in pointers:
#     pointer = [i,v]
#     pointer_new = dumpRotateImage(r, degree,pointer)
#     mask2[pointer_new[0],pointer_new[1]] = 255
#     img2[i,v] = img2[pointer_new[0],pointer_new[1]]
#     img2[pointer_new[0],pointer_new[1]] = img3[i,v]
#
# cv2.imshow("img2", img2)
# cv2.imwrite("../img2.png", img2)
# cv2.imshow("img3", img3)
# cv2.waitKey(0)
#
#
#
#
#
#
#
#
