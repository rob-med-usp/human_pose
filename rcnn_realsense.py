from src.RealSense import *
from src.KeypointRCNN import *
from src.Visualizer import *

import numpy as np

realsense = RealSense()
viz = Visualizer()
rcnn = KeypointRCNN()

realsense.initRealSense()
rcnn.defineModel()
viz.initWindows(Disparity=True)
viz.initPlot3D()

while True:

    frame, depth = realsense.getRealSenseFrames()

    persons = rcnn.predictFrame(frame)

    image = rcnn.drawSkeleton(frame, persons)
    
    if (len(persons)==0):
        viz.showImage(frame=frame)
        viz.showImage(depth, Disparity=True)
        continue
    
    viz.showImage(frame)
    viz.showImage(depth, Disparity=True)
    
    for person in persons:
        keypoints3D = realsense.deprojectPose3D(person)
        viz.plotPose3D(keypoints3D)
    
    viz.ax3D.clear()
    
    