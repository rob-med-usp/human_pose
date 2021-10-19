import src.HydraPose as hy
import cv2

hy.HydraPose(pose2D=hy.KEYPOINTMASKRCNN, pose3D=hy.REALSENSE)

hy.initRealsense()

while not hy.is_shutdown():
    
    persons = hy.run()

    persons[0]
