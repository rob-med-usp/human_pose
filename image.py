from src.HydraPose import HydraPose, SEFFPOSE
import cv2

hy = HydraPose(pose3D = SEFFPOSE)

path = '/media/guisoares/guisoares-ext-hdd/Datasets/camma_mvor_dataset/day1/cam1/color/'
fn = '000016.png'
file = path + fn
img = cv2.imread(file)

if img is None:
    print("No such file.")
    exit()
# img = cv2.imread("img/000014.png")
# img = cv2.imread("img3/test0.png")
cv2.imshow('Img', img)


persons = hy.estimate3DPose(img)

print(persons.shape)

hy.initWindow()
hy.plotPersons(img, block=True)

cv2.waitKey(1)