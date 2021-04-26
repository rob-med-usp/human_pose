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

    def __init__(self, pose2D_mode = "RCNN",fps = 60, resolution = (640,480), plot3D = True, ros_pub = True, from_bag_file = False, path_to_bag = ''):
        
        # Define keypoint mode
        self.pose2D_mode = pose2D_mode
        
        # Define plot mode
        self.plot3D = plot3D
        
        # Define communication mode
        self.ros_pub = ros_pub
        
        # Get stream from bag file
        self.from_bag_file = from_bag_file
        
        # Define edges
        self.pairs_OpenPose = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.pairs_RCNN = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.keypoints_names_COCO = ["nose","left_eye","right_eye","left_ear","rigth_ear","left_shoulder","right_shoulder","left_elbow","right_elbow",
                                "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        
        self.keypoints_names_OpenPose = ["nose","neck","left_shoulder","left_elbow","left_wrist","right_shoulder","right_elbow",
                                "right_wrist","left_hip","left_knee","left_ankle","right_hip","right_knee","right_ankle",
                                "left_eye","right_eye","left_ear","rigth_ear"]
        
        # Create windows
        self.win_name_rgb = 'RGB'
        self.win_name_depth = 'Disparity'
        
        cv2.namedWindow(self.win_name_rgb, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.win_name_depth, cv2.WINDOW_AUTOSIZE)
        
        cv2.moveWindow(self.win_name_rgb, 0, 0)
        cv2.moveWindow(self.win_name_depth, resolution[0], 0)
        
        # Default initializing RealSense
        if not self.from_bag_file:
            self.initRealSense(resolution = resolution, fps = fps)
            
        # Initialize stream from bag file
        if self.from_bag_file:
            self.initializeStreamFromBag(path_to_bag)
        
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
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

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
        
        return self.color_image_BGR, self.depth_image    
    
    def showFrames(self, block = False):
        
        cv2.imshow(self.win_name_rgb, self.color_image_BGR)
        cv2.imshow(self.win_name_depth, self.depth_colormap)
        
        if not block:
            if cv2.waitKey(1) == 27:
            
                quit()
        
        if block:
            cv2.waitKey() 
    
    def getPose3D(self, keypoints2D):
        
        self.keypoints2D = keypoints2D
        
        self.keypoints3D = np.ones((10, 17, 3)) * (-1)
        
        for id in range(len(keypoints2D)):
            for keypoint in range(len(keypoints2D[id])):
                if(keypoints2D[id][keypoint][0] != -1):
                    x = np.int(keypoints2D[id][keypoint][0])
                    y = np.int(keypoints2D[id][keypoint][1])
                    self.keypoints3D[id][keypoint] = rs.rs2_deproject_pixel_to_point(self.color_intrin, [x,y], self.depth_frame.get_distance(x, y))
                else:
                    self.keypoints3D[id][keypoint] = np.array([-1, -1, -1])
    
    def plotPose3DbyID(self, id):
        
        # Clear buff
        self.ax.clear()
        
        # TODO: eliminate this for
        x, y, z = [], [], []
        for keypoint in range(len(self.keypoints3D[id])):
            if(self.keypoints3D[id][keypoint][0] != -1):
                x.append(self.keypoints3D[id][keypoint][0])
                y.append(self.keypoints3D[id][keypoint][1])
                z.append(self.keypoints3D[id][keypoint][2])
        
        # Draw bones
        if(self.pose2D_mode == "OpenPose"):
            pairs = self.pairs_OpenPose
        if(self.pose2D_mode == "RCNN"):
            pairs = self.pairs_RCNN
            
        for id in range(len(self.keypoints3D)):
            for edges in pairs:
                
                if(self.keypoints3D[id][edges[0]] == [-1, -1] or self.keypoints3D[id][edges[1]] == [-1, -1]):
                    continue
                
                x1 = self.keypoints3D[id][edges[0]][0]
                y1 = self.keypoints3D[id][edges[0]][1]
                z1 = self.keypoints3D[id][edges[0]][2]

                x2 = self.keypoints3D[id][edges[1]][0]
                y2 = self.keypoints3D[id][edges[1]][1]
                z2 = self.keypoints3D[id][edges[1]][2]
                
                self.ax.plot([x1, x2],[z1, z2] ,[-y1, -y2])
        
        # Draw keypoints
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        self.ax.scatter(x, z, -y)
        
        # Display 3D plot
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)
    
    def setRosMessage(self, human_pose):
        
        human_pose.kpt0.x = self.keypoints3D[id][0][0]
        human_pose.kpt0.y = self.keypoints3D[id][0][1]
        human_pose.kpt0.z = self.keypoints3D[id][0][2]

        human_pose.kpt1.x = self.keypoints3D[id][1][0]
        human_pose.kpt1.y = self.keypoints3D[id][1][1]
        human_pose.kpt1.z = self.keypoints3D[id][1][2]
        
        human_pose.kpt2.x = self.keypoints3D[id][2][0]
        human_pose.kpt2.y = self.keypoints3D[id][2][1]
        human_pose.kpt2.z = self.keypoints3D[id][2][2]
        
        human_pose.kpt3.x = self.keypoints3D[id][3][0]
        human_pose.kpt3.y = self.keypoints3D[id][3][1]
        human_pose.kpt3.z = self.keypoints3D[id][3][2]
        
        human_pose.kpt4.x = self.keypoints3D[id][4][0]
        human_pose.kpt4.y = self.keypoints3D[id][4][1]
        human_pose.kpt4.z = self.keypoints3D[id][4][2]
        
        human_pose.kpt5.x = self.keypoints3D[id][5][0]
        human_pose.kpt5.y = self.keypoints3D[id][5][1]
        human_pose.kpt5.z = self.keypoints3D[id][5][2]
        
        human_pose.kpt6.x = self.keypoints3D[id][6][0]
        human_pose.kpt6.y = self.keypoints3D[id][6][1]
        human_pose.kpt6.z = self.keypoints3D[id][6][2]
        
        human_pose.kpt7.x = self.keypoints3D[id][7][0]
        human_pose.kpt7.y = self.keypoints3D[id][7][1]
        human_pose.kpt7.z = self.keypoints3D[id][7][2]
        
        human_pose.kpt8.x = self.keypoints3D[id][8][0]
        human_pose.kpt8.y = self.keypoints3D[id][8][1]
        human_pose.kpt8.z = self.keypoints3D[id][8][2]
        
        human_pose.kpt9.x = self.keypoints3D[id][9][0]
        human_pose.kpt9.y = self.keypoints3D[id][9][1]
        human_pose.kpt9.z = self.keypoints3D[id][9][2]
        
        human_pose.kpt10.x = self.keypoints3D[id][10][0]
        human_pose.kpt10.y = self.keypoints3D[id][10][1]
        human_pose.kpt10.z = self.keypoints3D[id][10][2]
        
        human_pose.kpt11.x = self.keypoints3D[id][11][0]
        human_pose.kpt11.y = self.keypoints3D[id][11][1]
        human_pose.kpt11.z = self.keypoints3D[id][11][2]
        
        human_pose.kpt12.x = self.keypoints3D[id][12][0]
        human_pose.kpt12.y = self.keypoints3D[id][12][1]
        human_pose.kpt12.z = self.keypoints3D[id][12][2]

        human_pose.kpt13.x = self.keypoints3D[id][13][0]
        human_pose.kpt13.y = self.keypoints3D[id][13][1]
        human_pose.kpt13.z = self.keypoints3D[id][13][2]

        human_pose.kpt14.x = self.keypoints3D[id][14][0]
        human_pose.kpt14.y = self.keypoints3D[id][14][1]
        human_pose.kpt14.z = self.keypoints3D[id][14][2]

        human_pose.kpt15.x = self.keypoints3D[id][15][0]
        human_pose.kpt15.y = self.keypoints3D[id][15][1]
        human_pose.kpt15.z = self.keypoints3D[id][15][2]

        human_pose.kpt16.x = self.keypoints3D[id][16][0]
        human_pose.kpt16.y = self.keypoints3D[id][16][1]
        human_pose.kpt16.z = self.keypoints3D[id][16][2]

        human_pose.kpt17.x = self.keypoints3D[id][17][0]
        human_pose.kpt17.y = self.keypoints3D[id][17][1]
        human_pose.kpt17.z = self.keypoints3D[id][17][2]

        return human_pose
    
    def publishOnROS(self, id = 0, TOPICLOGFLAG = False):
        
        if rospy.is_shutdown():
            print("Waiting for roscore")
            #TODO: flag de erro

        human_pose = Human_Pose()
        
        human_pose = self.setMessage(human_pose)
        
        if(TOPICLOGFLAG):
            #rospy.loginfo("I publish:")
            rospy.loginfo(human_pose)
            
        self.rosPub.publish(human_pose)
        self.rate.sleep()
        
    def printValues(self, id = 0, clearBuff = True):
        if clearBuff:
            os.system('clear')
        
        print('| Printing 2D (u, v) in pixels and 3D (x, y, z) in meters keypoints values: ')
        
        print(self.keypoints2D.shape)
        print(self.keypoints3D.shape)
        
        for body in range(len(self.keypoints3D[id])):
            #print('| ' + str(common.CocoPart(body)) + ' = ' + '(' + str(self.keypoints_2d[id][body][0]) + ', ' + str(self.keypoints_2d[id][body][1]) + 
            #     ')   (' + str(self.keypoints_3d[id][body][0]) + ', ' + str(self.keypoints_3d[id][body][1]) + ', ' + str(self.keypoints_3d[id][body][2]) +')')

            print('| {} = ({:.0f},{:.0f}) ({:.3f},{:.3f},{:.3f})'.format(self.keypoints_names_COCO[body],self.keypoints2D[id][body][0],self.keypoints2D[id][body][1],self.keypoints3D[id][body][0],self.keypoints3D[id][body][1],self.keypoints3D[id][body][2]))

    def initializeStreamFromBag(self, path_to_bag):
        
        if os.path.splitext(path_to_bag)[1] != '.bag':
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()
        
        # Create pipeline
        self.pipeline_bag = rs.pipeline()
        
        # Create a config object
        config_bag = rs.config()
        
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config_bag, path_to_bag)
        
        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        # TODO: configurable parameters
        config_bag.enable_stream(rs.stream.depth)
        config_bag.enable_stream(rs.stream.color)
        #config_bag.enable_all_streams()
        
        # Start streaming from file
        self.pipeline_bag.start(config_bag)
        
        # Create colorizer object
        self.colorizer_bag = rs.colorizer()
        
    def getBagFrames(self):
        
        # Get frame set
        frames = self.pipeline_bag.wait_for_frames()
        
        # Get depth and color frame
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()
        
        # Get color stream intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        
        # Colorize depth frame to jet colormap
        depth_colorized = self.colorizer_bag.colorize(self.depth_frame)
        
        # Get numpy format images
        self.depth_colormap = np.asanyarray(depth_colorized.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image_BGR = np.asanyarray(self.color_frame.get_data())
        
        return self.color_image_BGR, self.depth_image
            