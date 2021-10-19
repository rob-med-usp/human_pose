import numpy as np
import cv2

from src.OpenPose import * 
from src.KeypointRCNN import *
from src.SeffPose import *
from src.RealSense import *
from src.SkeletonsBridge import *
from src.Deproject import *
from src.Visualizer import *
from src.RosHandler import *

OPENPOSE = 0
KEYPOINTMASKRCNN = 1
SEFFPOSE = 2
REALSENSE = 3
FULL = 4

class HydraPose:

    def __init__(self, pose2D = KEYPOINTMASKRCNN, pose3D = FULL, ros = False):

        self.mode2D = pose2D
        self.mode3D = pose3D
        self.mode_ros = ros

        # Init architeture

        # Init pose 2D
        if self.mode2D == OPENPOSE:
            self.pose2d = OpenPose()
        if self.mode2D == KEYPOINTMASKRCNN:
            self.pose2d = KeypointRCNN()

        # Init bridge
        self.bridge = SkeletonsBridge()

        # Init deprojector
        if self.mode3D == FULL | self.mode3D == SEFFPOSE:
            self.deproj = Deprojector()

        # Init pose 3D
        if self.mode3D == FULL:
            self.seff = SeffPose()
            self.rls = RealSense()
            
        elif self.mode3D == SEFFPOSE:
            self.seff = SeffPose()

        elif self.mode3D == REALSENSE:
            self.rlsns = RealSense()

        # Init visualizer
        self.viz = Visualizer()

        # Init ROS
        if self.mode_ros == True:
            self.ros = RosHandler()
    
    def initWebcam(self, cam = 0):

        if self.mode3D != SEFFPOSE:
            print("Mode 3D doesn't work with webcam.")

        cap = cv2.VideoCapture(cam)

        return cap
    
    def initRealSense(self):

        return self.rlsns.initRealSense()

    def initVideoStream(self, filepath):

        if self.mode3D != SEFFPOSE:
            print("Mode 3D doesn't work with video stream.")
            exit()

        return cv2.VideoCapture(filepath)

    def getWebcamFrame(self, cap):

        ret, frame = cap.read()

        if ret is False:
            print("Failed to get frame.")
            exit()

        return frame
    
    def getRealSenseFrames(self, )


