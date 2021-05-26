from src.OpenPose import *
from src.Visualizer import *

fname = "img/000000.png"

pose2d = OpenPose()
viz = Visualizer()

pose2d.defineModel()
viz.initWindows()

image = viz.getImagefromFile(fname)

persons = pose2d.predictPose2D(image)

if len(persons>0):
    frame = pose2d.drawSkeleton(image)
    
viz.showImage(frame=image, block=True)

