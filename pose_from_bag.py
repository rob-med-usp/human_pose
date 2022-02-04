from src.HydraPose import HydraPose, SEFFPOSE, FULL
import os
import cv2
import numpy as np

hy = HydraPose(pose3D=FULL)

path = "/media/guisoares/guisoares-ext-hdd/Surgery-Records"
fn = "20211208_141701.bag"

abs_path = os.path.join(path,fn)

hy.initRealSenseBag(abs_path)

hy.initWindow()

while True:

    color, depth = hy.getRealSenseBagFrames()
    persons = hy.estimate3DPose(color)
    # hy.viz.drawSkeleton(color, hy.persons2D, upper_body= True)

    # persons3D = []
    # for person in hy.persons3DSeffPose:
    #     persons3D.append(person)
    # for person in hy.persons3DRealsense:
    #     persons3D.append(person)
    # persons3D = np.array(persons3D)

    # hy.viz.show(color, persons3D, block = False)
    hy.plotPersons(color, block = False)
    cv2.waitKey(1)