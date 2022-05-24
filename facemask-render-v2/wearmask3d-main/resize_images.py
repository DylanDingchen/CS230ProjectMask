import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import os, os.path
import cv2

import sys
sys.path.append('../')

file_path = "E:\SPR22\CS230_proj/facemask-render-v2\wearmask3d-main\src"
file_dst = "E:\SPR22\CS230_proj/facemask-render-v2\wearmask3d-main\src1"
dim = (256, 256)
  
# resize image

valid_images = [".jpg",".gif",".png",".tga"]
# print(os.listdir(file_path))
for f in os.listdir(file_path):
    
    fullPath = os.path.join(file_path, f)
    # print(fullPath)
    if (os.path.isdir(fullPath)):
        for subf in os.listdir(fullPath):
            imageP = os.path.join(fullPath,subf)
            # print(imageP)
            image = cv2.imread(imageP, 1)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(file_dst, subf), image)
            cv2.waitKey(0)
    #image = Image.open(fullPath, "r")

    #imgplot = plt.imshow(image)
    # if (os.path.isdir(fullPath)):
    #     for subf in os.listdir(fullPath):
    #         print(os.path.join(fullPath,subf))

