from src.HydraPose import HydraPose, SEFFPOSE
import cv2

hy = HydraPose(pose3D = SEFFPOSE)
cam = cv2.VideoCapture(0)

hy.initWindow()

while True:
    ret, frame = cam.read()

    persons3D = hy.estimate3DPose(frame)

    hy.plotPersons(frame, block=False)
