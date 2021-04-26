import classes.pose2D_rcnn as pose2D 

rcnn = pose2D.Pose2D_RCNN()

rcnn.defineModel()
rcnn.setWebcam(0)

while True:

    frame = rcnn.getWebcamFrame()
    outputs = rcnn.predictFrame(frame)
    image = rcnn.drawSkeleton(frame, outputs)
    rcnn.showImage(frame)



