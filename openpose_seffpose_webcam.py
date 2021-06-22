from src.OpenPose import * 
from src.SeffPose import *
from src.SkeletonsBridge import *
from src.Visualizer import *

pose2d = OpenPose()
pose3d = SeffPose()
bridge = SkeletonsBridge()
viz = Visualizer()


pose2d.defineModel(inWidth=200, inHeight=200)
pose3d.defineModel()
viz.initWindows()
viz.initPlot3D()

viz.setWebcam(0)

frame = viz.getWebcamFrame()
H = frame.shape[0]
W = frame.shape[1]
print('frame shape:' + '({},{})'.format(W,H))

while True:
    frame = viz.getWebcamFrame()
    persons = pose2d.predictPose2D(frame)

    if (len(persons) > 0):
        keypoints2D = bridge.OpenPoseCOCOtoCOCO(persons[0])
        keypoints2D = bridge.COCOtoMPII(keypoints2D)
        frame = viz.drawSkeleton(frame, keypoints2D)
            
        if (-1 not in persons[0]):
            keypoints2D_norm = pose3d.normalizePose2D(keypoints2D, W, H)
            keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)
            
            #TODO
            outputs = pose3d.getRawOutputs()
    
    viz.showImage(frame=frame)

    try:
        viz.plot3DHUMAN36(keypoints3D, outputs, block=False)
    except:
        pass