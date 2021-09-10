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
fname = 'img3/test4.png'
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
    
    t = time.time()
    keypoints3D = pose3d.estimatePose3Dfrom2DKeypoints(keypoints2D_norm)
    print("SeffPose time: {}".format(time.time()-t))
    
    t = time.time()
    kpts_global, imgpts, posepoints = deproj.deprojectPose(keypoints2D_HM36M, keypoints3D)
    print(f"Deprojection time: {time.time()-t}")
    
    image = viz.drawPersonAxis(image, np.array(keypoints2D_HM36M, dtype = np.uint16), imgpts)
    image = viz.drawSkeleton(image.copy(), keypoints2D_MPII, upper_body= True)
    
    for point in posepoints:
        point = point.ravel()
        print(point)
        image = cv2.circle(image, tuple(point.astype(np.int16)), 6, (0,0,255), thickness = -1)
                    
t = time.time()

cv2.namedWindow('Pose',cv2.WINDOW_FREERATIO)
cv2.imshow("Pose", image)

cv2.waitKey()

viz.showImage(frame = image)
viz.plotPose3D(kpts_global.T, block = True, upper_body = True, nose = False)
# viz.plotPose3D(keypoints3D, block = True, upper_body = True, nose = True)

print("Visualization time: {}".format(time.time()-t))
