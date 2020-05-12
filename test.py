# 强制使用CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# import
import numpy as np
import cv2
from time import time
from preprocess import preprocess

with open('filelist.txt','r') as f:
    imgs_src = f.read().split('\n')

imgs_src = np.asarray(imgs_src)
imgs_src = imgs_src.reshape((-1,3))

f = open('time.txt','w')

for IR_src, aligned_src, depth_src in imgs_src:
    # read imgs
    depth_img = cv2.imread(depth_src)
    IR_img = cv2.imread(IR_src)
    aligned_img = cv2.imread(aligned_src)

    # resize imgs
    # depth_img = cv2.resize(depth_img, (200,200))
    # IR_img = cv2.resize(IR_img, (200,200))
    # aligned_img = cv2.resize(aligned_img, (200,200))

    # preprocess
    start = time()
    preprocess(depth_img, IR_img, aligned_img)
    f.write(str(time() - start)+'\n')
