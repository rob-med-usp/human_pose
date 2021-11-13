from src.KeypointRCNN import * 
from src.Visualizer import *

pose2d = KeypointRCNN()
viz = Visualizer()

pose2d.defineModel()
viz.initWindows()

fname = "000000.png"

#image = viz.getImagefromFile(fname)
image = cv2.imread(fname)
keypoints, scores = pose2d.predictPose2D(image)
pose2d.drawSkeleton(image, keypoints, scores)
viz.showImage(frame=image, block=True)