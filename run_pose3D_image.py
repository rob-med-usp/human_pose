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

fname = "img/000000.png"

#image = viz.getImagefromFile(fname)
image = cv2.imread(fname)
persons, scores = pose2d.predictFrame(image)
pose2d.drawSkeleton(image, persons, scores)
keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(persons[0])
viz.plotPose3DbyID(0,keypoints3D)
viz.showImage(frame=image, block=True)