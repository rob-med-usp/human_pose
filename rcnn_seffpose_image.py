from src.OpenPose import * 
from src.KeypointRCNN import *
from src.SeffPose import *
from src.SkeletonsBridge import *
from src.Visualizer import *


pose2d = KeypointRCNN()
pose3d = SeffPose()
bridge = SkeletonsBridge()
viz = Visualizer()


pose2d.defineModel()
pose3d.defineModel(net = "GT")
viz.initWindows()
viz.initPlot3D()

# fname = "img/000004.png"
# fname = "single.jpeg"
# fname = 'img3/test6.png'
fname = 'img2/000296.jpg'

image = viz.getImagefromFile(fname)

H = image.shape[0]
W = image.shape[1]
print('Image shape:' + '({},{})'.format(W,H))

persons = pose2d.predictPose2D(image)

if (len(persons) > 0):
    keypoints2D_COCO = persons[0]
    keypoints2D_MPII = bridge.COCOtoMPII(keypoints2D_COCO)
    keypoints2D_HM36M = bridge.MPIItoHM36M(keypoints2D_MPII)
    keypoints2D_norm = pose3d.normalizePose2D(keypoints2D_HM36M, W, H)
    keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)
    image_openpose = viz.drawSkeleton(image.copy(), keypoints2D_MPII)
    #TODO
    outputs = pose3d.getRawOutputs()
    
viz.showImage(frame=image_openpose)

viz.plot3DHUMAN36(keypoints3D,outputs, block=True)
