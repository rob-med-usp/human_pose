import pyrealsense2 as rs
import time
import os

import cv2
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import rospy
from human_pose.msg import Human_Pose

class Pose3D_RealSense:

    def __init__(self, fps = 60, resolution = (640,480), self.plot3D = True, self.ros_pub = True):
        
        # Define edges
        self.edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        # Default initializing RealSense
        self.initRealSense(resolution = resolution, fps = fps)
        
        # Initialize Plot3D
        if self.plot3D:
            self.initPlot3D()
            
        if self.ros_pub:
            self.initROSPublisher()
        
    def initRealSense(self, resolution = (640,480), fps = 60):

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line)

        # Enable stream for RealSense
        res_x = resolution[0]
        res_y = resolution[1]
        self.config.enable_stream(rs.stream.depth, res_x, res_y, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, res_x, res_y, rs.format.bgr8, fps)
        
        self.pipeline.start(self.config)
        
        # Depth jetmap init
        self.colorizer = rs.colorizer()
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        #print(f"Device name: {}")
        
    def initPlot3D(self):
        
        # 3D plot init 
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection = '3d')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 4)
        self.ax.set_zlim(-2, 2)
        
    def initROSPublisher(self):
        
        # ROS Publisher init
        self.rosPub = rospy.Publisher("human_pose_id0", Human_Pose, queue_size = 54)
        # ROS node init
        rospy.init_node('human_pose_pub_node', anonymous=True)
        # Frequency init
        self.rate = rospy.Rate(30)
    
    def getRealSenseFrames(self):

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # Align frames
        frames = self.align.process(frames)

        # Get frames
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()

        # Validate frames
        while True:
            if not self.depth_frame or not self.color_frame:
                continue
            else:
                break

        # Convert images to numpy arrays
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image_BGR = np.asanyarray(self.color_frame.get_data())    
        
        # Get color stream intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics 

        # Aply jetmap to depth image for better visualization
        self.depth_colormap = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())    
        
    def showFrames(self):
        
        win_name_rgb = "RGB"
        win_name_depth = "Dsiparity"
        
        cv2.namedWindow(win_name_rgb, cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow(win_name_depth, cv2.WINDOW_GUI_EXPANDED)
        
        cv2.moveWindow(win_name_rgb, 0, 0)
        cv2.moveWindow(win_name_depth, self.color_image_BGR.shape[1], 0)
        
        cv2.imshow("RGB", self.color_image_BGR)
        cv2.imshow("Disparity", self.depth_image)
        
    def getPose3D(self, keypoints2D):
        
        self.keypoints3D = np.ones((10, 18, 3)) * (-1)
        
        for id in range(len(keypoints2D)):
            for keypoint in range(len(keypoints2D[id])):
                if(keypoints2D[id][keypoint] != [-1, -1]):
                    x = np.int(keypoints2D[id][body][0])
                    y = np.int(keypoints2D[id][body][1])
                    self.keypoints3D[id][body] = rs.rs2_deproject_pixel_to_point(self.color_intrin, [x,y], self.depth_frame.get_distance(x, y))
                else:
                    self.keypoints3D[id][body] = np.array([-1, -1, -1])
    
    def plotPose3DbyID(self, id):
        
        # TODO: eliminate this for
        x, y, z = [], [], []
        for keypoint in range(len(self.keypoints_3d[id])):
            if(self.keypoints_3d[id][body][0] != -1):
                x.append(self.keypoints_3d[id][body][0])
                y.append(self.keypoints_3d[id][body][1])
                z.append(self.keypoints_3d[id][body][2])
        
        # Draw bones
        for id in range(len(self.keypoints3D)):
            
                if keypoint == [-1, -1]

        
        