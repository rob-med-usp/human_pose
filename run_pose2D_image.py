import classes.pose2D_rcnn as pose2D 
import cv2
import os

images_path = "img/"

rcnn = pose2D.Pose2D_RCNN()

rcnn.defineModel()

for name in os.listdir(images_path):

    image = cv2.imread(images_path + name)
    
    outputs = rcnn.predictFrame(image)

    print(len(outputs[0]['keypoints']))
    print(outputs[0]['keypoints'])
    print(outputs[0]['labels'])
    print(outputs[0]['scores'])

    image = rcnn.drawSkeleton(image, outputs)

    #image = cv2.resize(image, (int(image.shape[1]*0.5),int(image.shape[0]*0.5)))
    rcnn.showImage(image, block = True)