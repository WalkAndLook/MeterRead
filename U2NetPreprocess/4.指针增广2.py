import cv2
import random
import numpy as np
import os

class Aug_image(object):
    # def __init__(self,image):
    #     self.image = image

    def img_resize(self,image, image_size):
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        # background[:, :, :] = 127
        background[:, :, :] = 0
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        image = background.copy()
        # print("image.shape:", image.shape)
        return image

    def dumpRotateImage(self, img, degree,pointer):
            '''
            输入参数：img--mask或者原图，degree--旋转的角度，pointer--要旋转的点
            '''
            height, width = img.shape[:2]
            # cv2.getRotationMatrix2D第一个参数为旋转中心坐标，这里取(1,1)，第二个为旋转角度取45，第三个为缩放比例
            # M = cv2.getRotationMatrix2D((height / 2width / 2,), degree, 1)
            # M = cv2.getRotationMatrix2D((216,192), degree, 1) # 油温
            M = cv2.getRotationMatrix2D((256,215), degree, 1) # 油温
            # M = cv2.getRotationMatrix2D((197,206), degree, 1) # sf61
            # M = cv2.getRotationMatrix2D((236,203), degree, 1) # sf62
            # M = cv2.getRotationMatrix2D((290,252), degree, 1) #  blq 0-3 倾斜
            # M = cv2.getRotationMatrix2D((208,208), degree, 1)  # blq 0-3
            # M = cv2.getRotationMatrix2D((315,204), degree, 1)  # blq 0-5
            # M = cv2.getRotationMatrix2D((212,205), degree, 1)  # ylb
            # 文件夹1，中心点为208,195，但是在OpenCV里面一般先行后列，所以是195,208
            #M = cv2.getRotationMatrix2D((195, 208), degree, 1)  # 1

            # 用这个矩阵来计算新的旋转的点
            new_pointer = [int(M[0][0] * pointer[0] + M[0][1] * pointer[1] + M[0][2]),
                       int(M[1][0] * pointer[0] + M[1][1] * pointer[1] + M[1][2])]

            # new_pointer = np.dot(M, np.array([[pointer[0]], [pointer[1]], [1]]))
            # print("new:",new_pointer)
            return new_pointer

    def rotate_mask(self,mask, degree,image,image2=None):
        # 像素值为255的都是指针所在位置，因为前面做了处理
        # 返回一个元组，第一个像素值是x行，第二个像素值是y列
        pointer_index = np.where(mask == 255)
        pointers = zip(pointer_index[0], pointer_index[1])
        mask = np.zeros(mask.shape).astype('uint8')
        for i, v in pointers:
            # mask的每个点的坐标
            pointer = [i, v]
            # 对mask上的点进行旋转
            pointer_new = self.dumpRotateImage(mask, degree, pointer)
            try:
                if image2 is None:
                        # 代表只需要mask的旋转
                        image[i, v] = image[pointer_new[0], pointer_new[1]]
                else:
                    # 代表需要原图的的指针旋转
                    image[i, v] = image2[pointer_new[0], pointer_new[1]]

                mask[pointer_new[0], pointer_new[1]] = 255
            except Exception as e:
                continue
        return image, mask

    def remove_noise(self,mask):
        pointer_index = np.where(mask == 255)
        pointers = zip(pointer_index[0], pointer_index[1])
        h, w = mask.shape
        for i, v in pointers:
            mask[min(i+1,w-1), min(v+1, h-1)] = 255
            mask[max(i-1,0), max(v-1,0)] = 255
            mask[min(i+1,w-1), max(v-1,0)] = 255
            mask[max(i-1,0), min(v+1,h-1)] = 255
        return mask

    def get_r_mask(self,mask):
        # 目标：得到图像的r通道
        kernel = np.ones((5, 5), np.uint8)
        # 如果指针本身不是很细的话，可以去掉膨胀
        # 对指针作膨胀操作
        mask = cv2.dilate(mask, kernel)
        # 图像通道进行拆分,返回bgr
        b_mask, g_mask, r_mask = cv2.split(mask)
        return  b_mask, g_mask, r_mask

    def aug_image(self,mask,image,degree):
        '''
        目标：指针旋转增强
        params：mask,image,degree--旋转的角度，一个数值
        '''
        b_mask, g_mask, r_mask = self.get_r_mask(mask)
        # 返回mask的r通道
        r_mask = np.where(r_mask>1,255,0).astype('uint8')
        # cv2.imshow("r_mask",r_mask)
        # cv2.waitKey(0)
        # 图像缩放416大小
        image = self.img_resize(image, 416)
        image2 = image.copy()

        # 返回mask旋转后的图像
        image_ratation, mask_rotation = self.rotate_mask(r_mask, degree, image)
        mask_rotation = self.remove_noise(mask_rotation)
        # 又做了一次翻转
        image_re_ratation, mask_re_rotation = self.rotate_mask(mask_rotation, -degree, image_ratation, image2)
        mask_rotation  = cv2.merge( [b_mask, g_mask,mask_rotation])

        # 最后只要第一次的mask翻转和做了两次翻转的原图像
        return image_re_ratation,mask_rotation

aug = Aug_image()

path = r'E:\codeWork\yibiao\meter(1)\crop\yibiao_mask\2'

count = 0
for file in os.listdir(path):
    if ".jpg" in file :
        image_path = os.path.join(path,file)
        mask_path = image_path.replace(".jpg",".png")
        filepath,fullflname = os.path.split(image_path)
        fname,ext = os.path.splitext(fullflname)

        # 图像和mask缩放到416
        mask = cv2.imread(mask_path,1)
        mask = aug.img_resize(mask,416)
        image = cv2.imread(image_path,1)
        image = aug.img_resize(image,416)

        # 设置旋转角度
        # degree = np.linspace(0,270,50) #旋转角度
        # degree = np.linspace(0,100,10) #旋转角度 避雷器0-3 0-5
        # degree = np.linspace(0,240,15) #旋转角度 压力表
        # degree = np.linspace(-0,90,5) #旋转角度 油温
        # degree = np.linspace(0,220,15) #旋转角度 避雷器sf61
        # degree = np.linspace(0,80,8) #旋转角度 避雷器sf61
        degree = np.linspace(-100, 20, 5)  # 旋转角度 1
        for i in degree:
            # 对旋转角度做一个随机扰动
            i = i+ random.randint(-5,5)
            i = int(i)
            print(mask.shape)
            # 做指针旋转增强
            img_aug, mask_aug= aug.aug_image(mask,image,i)


            cv2.imwrite(r"{}\{}_{}.png".format(filepath,fname,count),mask_aug)
            cv2.imwrite(r"{}\{}_{}.jpg".format(filepath,fname,count),img_aug)
            count+=1






