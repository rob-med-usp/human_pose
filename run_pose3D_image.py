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

#fname = "img/000079.png"
fname = "single.jpeg"

image = viz.getImagefromFile(fname)

persons, scores = pose2d.predictFrame(image)

keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(persons[0][:,[0,1]])
keypoints2D = pose3d.getMPIIKeypoints()

pose2d.drawSkeleton(image, persons, scores)
image = viz.drawSkeleton(image, keypoints2D)

outputs = pose3d.getRawOutputs()
viz.showImage(frame=image, block=True)
viz.plot3DHUMAN36(keypoints3D,outputs)
