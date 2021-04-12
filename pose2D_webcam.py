import pose2D_rcnn as pose2D

pose2D = pose2D.Pose2D_RCNN()

pose2D.defineModel()

pose2D.setWebcam(0)

while True:

    ret, frame = pose2D.getWebcamFrame()

    if ret:
        outputs = pose2D.predictFrame(frame)
        image = pose2D.drawSkeleton(frame, outputs)
    
        pose2D.showImage(frame)



