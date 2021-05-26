from src.KeypointRCNN import * 
from src.Visualizer import *
from src.SeffPose import *

pose2d = KeypointRCNN()
viz = Visualizer()
pose3d = SeffPose()

pose2d.defineModel()
pose3d.defineModel()
viz.initWindows()
viz.initPlot3D()

fname = "img/000057.png"
#fname = "single.jpeg"

image = viz.getImagefromFile(fname)

persons = pose2d.predictFrame(image)
if (len(persons) > 0):
    keypoints2D = pose3d.COCOtoMPII(persons[0])
    keypoints2D_norm = pose2d.normalizeKeypoints(image, keypoints2D)
    keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)

    pose2d.drawSkeleton(image, persons)
    image = viz.drawSkeleton(image, keypoints2D)
    #TODO
    outputs = pose3d.getRawOutputs()
    
viz.showImage(frame=image)
viz.plot3DHUMAN36(keypoints3D,outputs)
