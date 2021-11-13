from src import HydraPose
import cv2

hy = HydraPose.HydraPose(pose3D = HydraPose.SEFFPOSE)
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    persons3D = hy.estimate3DPose(frame)

    print(persons3D)
