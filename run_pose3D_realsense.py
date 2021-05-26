import classes.pose2D_rcnn as pose2D
import classes.pose3D_realsense as pose3D

import numpy as np

rcnn = pose2D.Pose2D_RCNN()
realsense = pose3D.Pose3D(ros_pub = False)

rcnn.defineModel()

while True:

    frame, depth = realsense.getRealSenseFrames()

    outputs = rcnn.predictFrame(frame)

    image = rcnn.drawSkeleton(frame, outputs)
    
    keypoints2D, person_scores = rcnn.getKeypointsAndPersonScores()
    
    realsense.deprojectPose3D(keypoints2D)
    
    realsense.showFrames()
    
    realsense.printValues()
    
    realsense.plotPose3DbyID(0)