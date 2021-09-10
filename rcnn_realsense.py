from src.RealSense import *
from src.KeypointRCNN import *
from src.SkeletonsBridge import *
from src.Visualizer import *

import numpy as np

realsense = RealSense()
bridge = SkeletonsBridge()
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
    
    for person in persons:
        t = time.time()
        
        keypoints2D_MPII = bridge.COCOtoMPII(person)
        print(keypoints2D_MPII)
        if len(keypoints2D_MPII)!=0:
            image = viz.drawSkeleton(frame, keypoints2D_MPII, upper_body=True)
        keypoints3D = realsense.deprojectPose3D(keypoints2D_MPII)
        # print(keypoints3D)
        print(f"Deproj time: {time.time()-t}")
    
    if (len(persons)==0):
        viz.showImage(frame=frame)
        # viz.showImage(depth, Disparity=True)
        continue
    
    viz.showImage(image)
    viz.showDepthImage(depth)
    # viz.showImage(depth, Disparity=True)
    t = time.time()
    if len(keypoints3D) != 0:
        viz.plotRealSenseDeproj(keypoints3D, block=False, upper_body=True)
    print(f'Plot time: {time.time()-t}')
    viz.ax3D.clear()