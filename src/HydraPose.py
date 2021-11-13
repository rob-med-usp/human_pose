import numpy as np
import cv2

from src.OpenPose import * 
from src.KeypointRCNN import *
from src.SeffPose import *
from src.RealSense import *
from src.SkeletonsBridge import *
from src.Deproject import *
from src.Fusion import *
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

        #OS type
        # Check if OS is win or not
        self.is_windows = sys.platform.startswith('win')
        
        # Init architeture
        # Init pose 2D
        if self.mode2D == OPENPOSE:
            self.pose2d = OpenPose()
        if self.mode2D == KEYPOINTMASKRCNN:
            self.pose2d = KeypointRCNN()

        self.pose2d.defineModel()

        # Init bridge
        self.bridge = SkeletonsBridge()

        # Init deprojector
        if self.mode3D == FULL or self.mode3D == SEFFPOSE:
            self.deproj = Deprojector()

        # Init pose 3D
        if self.mode3D == FULL:
            self.seff = SeffPose()
            self.rlsns = RealSense()
            self.fus = Fusion()
            
        elif self.mode3D == SEFFPOSE:
            self.seff = SeffPose()
            self.seff.defineModel()

        elif self.mode3D == REALSENSE:
            self.rlsns = RealSense()

        # Init visualizer
        self.viz = Visualizer()

        # Init ROS
        if self.mode_ros == True:
            self.ros = RosHandler()

    def initRealSense(self):
        self.rlsns.initRealSense()

    def getRealSenseFrames(self):
        
        self.color_img, self.depth_img = self.rlsns.getRealSenseFrames()
        
        return self.color_img, self.depth_img

    def initRealSenseBag(self, filepath):
        self.rlsns.initializeStreamFromBag(filepath)

    def getRealSenseBagFrames(self):

        self.color_img, self.depth_img = self.rlsns.getBagFrames()

        return self.color_img, self.depth_img
    
    def estimate3DPose(self, color_img):

        if(color_img is None):
            print("Image empty.")
            return


        h = color_img.shape[0]
        w = color_img.shape[1]

        self.persons2D = self.pose2d.estimate2DPose(color_img)

        # Return if persons is empty
        if len(self.persons2D) == 0:
            return []
        
        if self.mode3D == FULL:
            persons3DRealsense = self.estimate3DPoseRealsense(self.persons2D)
            persons3DSeffPose = self.estimate3DPoseSeffpose(self.persons2D, w, h)
            self.persons3DHybrid = self.fuseResultsSeffPoseRealsense(persons3DRealsense, persons3DSeffPose)

        elif self.mode3D == SEFFPOSE:
            persons3DSeffPose = self.estimate3DPoseSeffpose(self.persons2D, w, h)
            self.persons3DHybrid = persons3DSeffPose 

        elif self.mode3D == REALSENSE:
            persons3DRealsense = self.estimate3DPoseRealsense(self.persons2D)
            self.persons3DHybrid = persons3DRealsense
        
        return self.persons3DHybrid

    def estimate3DPoseRealsense(self, persons2D):
        
        persons3D = []
        for person2D in persons2D:
            person3D = self.rlsns.deprojectPose3D(person2D)
            persons3D.append(person3D)
        
        return np.array(persons3D)

    def estimate3DPoseSeffpose(self, persons2D, w, h):
        
        persons3D = []
        for person2D in persons2D:
            person2D_norm = self.seff.normalizePose2D(person2D, w, h)
            person3D = self.seff.estimatePose3Dfrom2DKeypoints(person2D_norm)
            person3D_deproj,_,_ = self.deproj.deprojectPose(person2D, person3D)
            persons3D.append(person3D_deproj)
        
        return np.array(persons3D)
    
    def fuseResultsSeffPoseRealsense(self, persons3DRealsense, persons3DSeffPose):
        
        if(persons3DSeffPose.shape != persons3DRealsense.shape):
            print("Results shape dont match!")
        
        personsHybrid3D = []
        for i in range(len(persons3DSeffPose)):
            person3D = self.fus.mergeResults(persons3DRealsense[i], persons3DSeffPose[i])
            personsHybrid3D.append(person3D)
        
        return np.array(personsHybrid3D)

    def initWindow(self):
        self.viz.initWindows()

    def plotPersons(self, color_img, block = True):

        self.viz.drawSkeleton(color_img, self.persons2D, upper_body= True)
        self.viz.show(color_img, self.persons3DHybrid, block = block)
        # self.viz.show(image,self.depth_img,self.persons3DHybrid, block = False)
