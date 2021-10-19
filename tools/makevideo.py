import cv2
import numpy as np
import os

path ='img2/'
list = os.listdir(path)
list.sort()
print(list)
img_array = []
for filename in list:
    img = cv2.imread(path + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('simple.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()