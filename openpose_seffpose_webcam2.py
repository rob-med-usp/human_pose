from src.OpenPose import *
from src.SeffPose import *
from src.SkeletonsBridge import *
from src.Visualizer import *

fname = "img/000000.png"

pose2d = OpenPose()
bridge = SkeletonsBridge()
pose3d = SeffPose()
viz = Visualizer()

pose2d.defineModel()
pose3d.defineModel()
viz.initWindows()
viz.initPlot3D()
viz.setWebcam(0)

while True:
    
    frame = viz.getWebcamFrame()
    
    # Predict pose 2D with OpenPose
    persons = pose2d.predictPose2D(frame)
    
    # If there is no persons in image, run to next iteration
    if (len(persons)==0):
        viz.showImage(frame=frame, block=False)
        continue
    print(persons[0])
    # Get first person of image
    keypoints2D_OpenPoseCOCO = persons[0]
    frame = pose2d.drawSkeleton(frame)
    viz.showImage(frame=frame, block=False)
    
    if(-1 in keypoints2D_OpenPoseCOCO):
        continue
    
    # Adapt OpenPose keypoint format to SeffPose
    keypoints2D_COCO = bridge.OpenPoseCOCOtoCOCO(keypoints2D_OpenPoseCOCO)
    keypoints2D_MPI = bridge.COCOtoMPII(keypoints2D_COCO)
    keypoints2D_norm = bridge.normalizeKeypoints(frame, keypoints2D_MPI)
    
    # Predict pose 3D with SeffPose
    keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)
    outputs = pose3d.getRawOutputs()
    
    viz.plot3DHUMAN36(keypoints3D, outputs)
    
    