from src.HydraPose import HydraPose, SEFFPOSE
import cv2

hy = HydraPose(pose3D = SEFFPOSE)

img = cv2.imread("img/000025.png")
# img = cv2.imread("img3/test0.png")

persons = hy.estimate3DPose(img)

print(persons.shape)

hy.initWindow()
hy.plotPersons(img, block=True)