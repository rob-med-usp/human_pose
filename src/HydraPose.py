import numpy as np
import cv2
from enum import Enum

from .OpenPose import * 
from .KeypointRCNN import *
from .SeffPose import *
from .RealSense import *
from .SkeletonsBridge import *
from .Deproject import *
from .Fusion import *
from .Visualizer import *
from .RosHandler import *

OPENPOSE = 0
KEYPOINTMASKRCNN = 1
SEFFPOSE = 2
REALSENSE = 3
FULL = 4

class HydraPose:

    def __init__(self, pose2D = KEYPOINTMASKRCNN, pose3D = FULL, ros = False):

        # Config atribs
        self.mode2D = pose2D
        self.mode3D = pose3D
        self.mode_ros = ros

        #OS type
        # Check if OS is win or not
        self.is_windows = sys.platform.startswith('win')

        # Standard atribs
        self.persons2D = None
        self.persons3DHybrid  = None

        
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
            self.seff.defineModel()

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
    
    def setIntrinsics(self, intrinsics, distortion):
        
        assert(type(intrinsics).__module__ == np.__name__, "Instrinsics input must be a numpy array.")
        assert(type(distortion).__module__ == np.__name__, "Distortion input must be a numpy array.")

        self.deproj.intrinsics = intrinsics
        self.deproj.distortion = distortion
    
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
            self.persons3DRealsense = self.estimate3DPoseRealsense(self.persons2D)
            self.persons3DSeffPose = self.estimate3DPoseSeffpose(self.persons2D, w, h)
            self.persons3DHybrid = self.fuseResultsSeffPoseRealsense(self.persons3DRealsense, self.persons3DSeffPose)

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
        # Meters to milimiter
        persons3D = np.array(persons3D)*1000
        
        return persons3D

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

    def plotPersons(self, color_img, mode='Human36M', block = True):

        self.viz.drawSkeleton(color_img, self.persons2D, mode=mode, upper_body=True)
        self.viz.show(color_img, self.persons3DHybrid, block=block, mode=mode)
        # self.viz.show(image,self.depth_img,self.persons3DHybrid, block = False)

class Person:

    def __init__(self, kpts2D = None, kpts3D = None):
        self.kpts2D = kpts2D
        self.kpts3D = kpts3D
        self.frame = Frame.WORLD
        self.id = None
        self.name = None
    
    def transformFrame(self, frameTo):
        pass 

    def toSkeletonType(self):
        pass

class Frame(Enum):

    WORLD = 0
    ROBOT = 1
    CAM = 2