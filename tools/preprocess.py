
import cv2
from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import os 

def max_gray(image):
    # 最大灰度化方法
    b, g, r = cv2.split(image)
    index_1 = b > g
    result = np.where(index_1, b, g)
    index_2 = result > r
    result = np.where(index_2, result, r)
    return result

def mean_gray(image):
    # 平均灰度化方法
    return np.mean(image, axis=-1)

def weight_gray(image):
    # 加权平均灰度化方法
    w = cfg.RGB_WEIGHTS
    image = np.sum(image*w, axis=-1)
    return image

def enhance_value_max(image, gamma=1.6, norm=True):
    image = cv2.medianBlur(image, 3)
    image = mean_gray(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = enhance(image, gamma=gamma, norm=norm)
    return dst

def normalize(a):
    # 归一化
    amin, amax = a.min(), a.max()  # 求最大最小值
    result = (a-amin)/(amax-amin)  # (矩阵元素-最小值)/(最大值-最小值)
    return result

def enhance(image, gamma=1.6, norm=True):
    # 伽马变换
    fI = (image+0.5)/256
    fI = fI + np.median(fI)
    dst = np.power(fI, gamma)
    
    if norm:
        dst = normalize(dst)        

    return dst

def image_process(image):
    image = enhance_value_max(image, gamma=1.6, norm=False)
    image += ercode_dilate(image, 10) 
    # 为什么是加呢，因为去掉干扰线的同时也可能去掉字母比较细的地方
    return image

def ercode_dilate(img, threshold):
    # 腐蚀参数， (threshold, threshold)为腐蚀矩阵大小
    kernel = np.ones((threshold, threshold), np.uint8)
    # 腐蚀图片
    img = cv2.dilate(img, kernel, iterations=5)
    # 膨胀图片
    img = cv2.erode(img, kernel, iterations=3)
    return img


cm = get_cmap("gray")
path = './dataset2/train'
out_path = './dcic/predTrain'
files = os.listdir(path)
for file in files:
    image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
    en_image = image_process(image)
    norm = Normalize(vmin=en_image.min(), vmax=en_image.max())
    img2 = cm(norm(en_image), bytes=True)
    cv2.imwrite(os.path.join(out_path, file.replace('png','jpg')), img2)
    

