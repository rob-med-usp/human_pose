from src.OpenPose import *
from src.Visualizer import *

fname = "img/000000.png"

pose2d = OpenPose()
viz = Visualizer()

pose2d.defineModel()
viz.initWindows()
viz.setWebcam(0)

while True:
    
    frame = viz.getWebcamFrame()
        
    persons = pose2d.predictPose2D(frame)
    
    if len(persons>0):
        frame = pose2d.drawSkeleton(frame)
    viz.showImage(frame=frame)