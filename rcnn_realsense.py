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

    t = time.time()
    persons = rcnn.predictPose2D(frame)
    print(f"MaskRCNN time: {time.time()-t}")
    
    image = rcnn.drawSkeleton(frame, persons)
    
    if (len(persons)==0):
        viz.showImage(frame=frame)
        # viz.showImage(depth, Disparity=True)
        continue
    
    viz.showImage(frame)
    # viz.showImage(depth, Disparity=True)
    
    for person in persons:
        t = time.time()
        keypoints3D = realsense.deprojectPose3D(person)
        # print(keypoints3D)
        print(f"Deproj time: {time.time()-t}")
        
        t = time.time()
        viz.plotPose3D(keypoints3D, block=False)
        print(f'Plot time: {time.time()-t}')
    viz.ax3D.clear()
    
    