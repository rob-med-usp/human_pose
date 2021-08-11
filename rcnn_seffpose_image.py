from src.OpenPose import * 
from src.KeypointRCNN import *
from src.SeffPose import *
from src.SkeletonsBridge import *
from src.Deproject import *
from src.Visualizer import *

import matplotlib.pyplot as plt

pose2d = KeypointRCNN()
pose3d = SeffPose()
bridge = SkeletonsBridge()
deproj = Deprojector()
viz = Visualizer()


pose2d.defineModel()
pose3d.defineModel(net = "GT")
viz.initWindows()
viz.initPlot3D()

# fname = "img/000069.png"
# fname = 'img/000013demo.png'
# fname = "single.jpeg"
fname = 'img3/test6.png'
# fname = 'img2/000296.jpg'

image = viz.getImagefromFile(fname)

H = image.shape[0]
W = image.shape[1]
print('Image shape:' + '({},{})'.format(W,H))

t = time.time()
persons = pose2d.predictPose2D(image)
print("RCNN time: {}".format(time.time()-t))

print(len(persons))
if (len(persons) > 0):
    keypoints2D_COCO = persons[0]
    
    t = time.time()
    keypoints2D_MPII = bridge.COCOtoMPII(keypoints2D_COCO)
    keypoints2D_HM36M = bridge.MPIItoHM36M(keypoints2D_MPII)
    keypoints2D_norm = pose3d.normalizePose2D(keypoints2D_HM36M, W, H)
    print("Skeleton operations time: {}".format(time.time()-t))
    
    
    keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)
    print("SeffPose time: {}".format(time.time()-t))
    
    # kpts_global, rvecs, tvecs, imgpts = deproj.deprojectPose(keypoints2D_MPII, keypoints3D)
    
    # print(kpts_global)
    # print(tvecs)
    
    t = time.time()
    image = viz.drawSkeleton(image.copy(), keypoints2D_MPII, upper_body= True)
    
t = time.time()
viz.showImage(frame = image)
# viz.plotPose3D(kpts_global, block = True, upper_body = False)
viz.plotPose3D(keypoints3D, block = True, upper_body = True, nose = True)
# viz.plot3DHUMAN36(keypoints3D,outputs, block=True)
print("Visualization time: {}".format(time.time()-t))
