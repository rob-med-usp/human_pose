from src.KeypointRCNN import * 
from src.Visualizer import *

pose2d = KeypointRCNN()
viz = Visualizer()

pose2d.defineModel()

viz.initWindows()
viz.setWebcam(0)

while True:
    frame = viz.getWebcamFrame()
    keypoints, scores = pose2d.predictFrame(frame)
    pose2d.drawSkeleton(frame, keypoints, scores)
    viz.showImage(frame=frame)



