from src.OpenPose import *
from src.Visualizer import *
import time

fname = "img/000000.png"

pose2d = OpenPose()
viz = Visualizer()

pose2d.defineModel()
viz.initWindows()

image = viz.getImagefromFile(fname)

t = time.time()
persons = pose2d.predictPose2D(image)
print(f"OpenPose time: {1/(time.time()-t)}")

print(persons)

if len(persons>0):
    frame = pose2d.drawSkeleton(image)
    
viz.showImage(frame=image, block=True)

