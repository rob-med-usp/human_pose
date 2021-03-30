import argparse
import time
import os

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

import pyrealsense2 as rs 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import rospy
from human_pose.msg import Human_Pose

class RsHumanPose:

    # Model params
    def __init__(self, args):

        def str2bool(v):
            return v.lower() in ("yes", "true", "t", "1")
        
        self.w, self.h = model_wh(args.resize)
    
        # Model construction
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(args.model), target_size=(self.w, self.h), trt_bool=str2bool(args.tensorrt))
        else:
            self.e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
    
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Enable stream for RealSense
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

        self.pipeline.start(self.config)

        #self.pc = rs.pointcloud()
        
        # Depth jetmap init
        self.colorizer = rs.colorizer()
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 3D plot init 
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection = '3d')

        # ROS Publisher init
        self.rosPub = rospy.Publisher("human_pose_id0", Human_Pose, queue_size = 54)
        # ROS node init
        rospy.init_node('human_pose_pub_node', anonymous=True)
        # Frequency init
        self.rate = rospy.Rate(30)

    def getPose2D(self):
        # initialize 2d keypoints vector
        keypoints_2d_norm = np.ones((10,18,2))*-1
        self.keypoints_2d = np.ones((10,18,2))*-1

        humans_iter = 0    
        for human in self.humans:
            for i in range(common.CocoPart.Background.value):

                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                keypoints_2d_norm[humans_iter][i][0] = body_part.x 
                keypoints_2d_norm[humans_iter][i][1] = body_part.y
            humans_iter += 1

        for id in range(len(keypoints_2d_norm)):
            for body in range(len(keypoints_2d_norm[id])):
                if (keypoints_2d_norm[id][body][0] != -1): 
                    self.keypoints_2d[id][body][0] = keypoints_2d_norm[id][body][0] * self.color_image_BGR.shape[1] 
                    self.keypoints_2d[id][body][1] = keypoints_2d_norm[id][body][1] * self.color_image_BGR.shape[0]

        return self.keypoints_2d

    def getPose3D(self):
        self.keypoints_3d = np.ones((10,18,3))

        for id in range(len(self.keypoints_2d)):
            for body in range(len(self.keypoints_2d[id])):
                if(self.keypoints_2d[id][body][0] != -1 and self.keypoints_2d[id][body][1] != -1):
                    x = np.int(self.keypoints_2d[id][body][0])
                    y = np.int(self.keypoints_2d[id][body][1])
                    self.keypoints_3d[id][body] = rs.rs2_deproject_pixel_to_point(self.color_intrin, [x,y], self.depth_frame.get_distance(x, y))
                else:
                    self.keypoints_3d[id][body] = np.array([-1,-1,-1])
        
        return self.keypoints_3d
    
    def updatePose(self):
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

        #mapped_frame = self.color_frame
        #points = self.pc.calculate(self.depth_frame)
        #self.pc.map_to(mapped_frame)

        # Get color stream intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics 

        # Aply jetmap to depth image for better visualization
        self.depth_colormap = np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())

        # OpenPose TF1.5.1 (TODO: keras implementation)
        self.humans = self.e.inference(self.color_image_BGR, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=args.resize_out_ratio)

        # Update keypoints arrays
        self.getPose2D()
        self.getPose3D()

    def printValues(self, id = 0, clearBuff = True):
        if clearBuff:
            os.system('clear')
        print('| Printing 2D (u, v) in pixels and 3D (x, y, z) in meters keypoints values: ')
        for body in range(18):
            #print('| ' + str(common.CocoPart(body)) + ' = ' + '(' + str(self.keypoints_2d[id][body][0]) + ', ' + str(self.keypoints_2d[id][body][1]) + 
            #     ')   (' + str(self.keypoints_3d[id][body][0]) + ', ' + str(self.keypoints_3d[id][body][1]) + ', ' + str(self.keypoints_3d[id][body][2]) +')')

            print('| {} = ({:.0f},{:.0f}) ({:.3f},{:.3f},{:.3f})'.format(common.CocoPart(body),self.keypoints_2d[id][body][0],self.keypoints_2d[id][body][1],self.keypoints_3d[id][body][0],self.keypoints_3d[id][body][1],self.keypoints_3d[id][body][2]))
   
    def drawAndDisplay(self):
            self.color_image_BGR = TfPoseEstimator.draw_humans(self.color_image_BGR, self.humans, imgcopy=False)
                                
            # show FPS on image
            # cv2.putText(self.color_image_BGR,
                        # "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        # (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        # (0, 255, 0), 2)

            # Display images
            cv2.imshow('color image BGR', self.color_image_BGR)
            cv2.imshow('depth colorized', self.depth_colormap)
            # fps_time = time.time()
            # if cv2.waitKey(1) == 27:
            #     break

    def plotPose3D(self, id = 0):

        self.ax.clear()
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 4)
        self.ax.set_zlim(-2, 2)

        x, y, z = [], [], []
        for body in range(len(self.keypoints_3d[id])):
            if(self.keypoints_3d[id][body][0] != -1):
                x.append(self.keypoints_3d[id][body][0])
                y.append(self.keypoints_3d[id][body][1])
                z.append(self.keypoints_3d[id][body][2])

        
        # Draw bones
        for human in self.humans:
            for pair_order, pair in enumerate(common.CocoPairsRender):

                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                x1 = self.keypoints_3d[id][pair[0]][0]
                y1 = self.keypoints_3d[id][pair[0]][1]
                z1 = self.keypoints_3d[id][pair[0]][2]

                x2 = self.keypoints_3d[id][pair[1]][0]
                y2 = self.keypoints_3d[id][pair[1]][1]
                z2 = self.keypoints_3d[id][pair[1]][2]

                self.ax.plot([x1, x2],[z1, z2] ,[-y1, -y2])

        # filtered_kpts = np.array([x,y,z])
        
        # rotation_x = np.array([[1, 0, 0 ],
        #                        [0, 0, -1],
        #                        [0, 1, 0 ]])

        # if filtered_kpts.shape[1] != 0:
        #     rotated_kpts = np.dot(rotation_x, filtered_kpts)

        #     print(rotated_kpts.shape)
        #     x = rotated_kpts[:,0][:]
        #     y = rotated_kpts[:,1][:]
        #     z = rotated_kpts[:,2][:]

            # Draw keypoints
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        self.ax.scatter(x, z, -y)
        #self.ax.scatter(x, y, z)

        # Display 3D plot
        plt.draw()
        plt.show(block=False)
        plt.pause(0.001)

        #plt.show()

    def setMessage(self, human_pose, id=0):

        human_pose.kpt0.x = self.keypoints_3d[id][0][0]
        human_pose.kpt0.y = self.keypoints_3d[id][0][1]
        human_pose.kpt0.z = self.keypoints_3d[id][0][2]

        human_pose.kpt1.x = self.keypoints_3d[id][1][0]
        human_pose.kpt1.y = self.keypoints_3d[id][1][1]
        human_pose.kpt1.z = self.keypoints_3d[id][1][2]
        
        human_pose.kpt2.x = self.keypoints_3d[id][2][0]
        human_pose.kpt2.y = self.keypoints_3d[id][2][1]
        human_pose.kpt2.z = self.keypoints_3d[id][2][2]
        
        human_pose.kpt3.x = self.keypoints_3d[id][3][0]
        human_pose.kpt3.y = self.keypoints_3d[id][3][1]
        human_pose.kpt3.z = self.keypoints_3d[id][3][2]
        
        human_pose.kpt4.x = self.keypoints_3d[id][4][0]
        human_pose.kpt4.y = self.keypoints_3d[id][4][1]
        human_pose.kpt4.z = self.keypoints_3d[id][4][2]
        
        human_pose.kpt5.x = self.keypoints_3d[id][5][0]
        human_pose.kpt5.y = self.keypoints_3d[id][5][1]
        human_pose.kpt5.z = self.keypoints_3d[id][5][2]
        
        human_pose.kpt6.x = self.keypoints_3d[id][6][0]
        human_pose.kpt6.y = self.keypoints_3d[id][6][1]
        human_pose.kpt6.z = self.keypoints_3d[id][6][2]
        
        human_pose.kpt7.x = self.keypoints_3d[id][7][0]
        human_pose.kpt7.y = self.keypoints_3d[id][7][1]
        human_pose.kpt7.z = self.keypoints_3d[id][7][2]
        
        human_pose.kpt8.x = self.keypoints_3d[id][8][0]
        human_pose.kpt8.y = self.keypoints_3d[id][8][1]
        human_pose.kpt8.z = self.keypoints_3d[id][8][2]
        
        human_pose.kpt9.x = self.keypoints_3d[id][9][0]
        human_pose.kpt9.y = self.keypoints_3d[id][9][1]
        human_pose.kpt9.z = self.keypoints_3d[id][9][2]
        
        human_pose.kpt10.x = self.keypoints_3d[id][10][0]
        human_pose.kpt10.y = self.keypoints_3d[id][10][1]
        human_pose.kpt10.z = self.keypoints_3d[id][10][2]
        
        human_pose.kpt11.x = self.keypoints_3d[id][11][0]
        human_pose.kpt11.y = self.keypoints_3d[id][11][1]
        human_pose.kpt11.z = self.keypoints_3d[id][11][2]
        
        human_pose.kpt12.x = self.keypoints_3d[id][12][0]
        human_pose.kpt12.y = self.keypoints_3d[id][12][1]
        human_pose.kpt12.z = self.keypoints_3d[id][12][2]

        human_pose.kpt13.x = self.keypoints_3d[id][13][0]
        human_pose.kpt13.y = self.keypoints_3d[id][13][1]
        human_pose.kpt13.z = self.keypoints_3d[id][13][2]

        human_pose.kpt14.x = self.keypoints_3d[id][14][0]
        human_pose.kpt14.y = self.keypoints_3d[id][14][1]
        human_pose.kpt14.z = self.keypoints_3d[id][14][2]

        human_pose.kpt15.x = self.keypoints_3d[id][15][0]
        human_pose.kpt15.y = self.keypoints_3d[id][15][1]
        human_pose.kpt15.z = self.keypoints_3d[id][15][2]

        human_pose.kpt16.x = self.keypoints_3d[id][16][0]
        human_pose.kpt16.y = self.keypoints_3d[id][16][1]
        human_pose.kpt16.z = self.keypoints_3d[id][16][2]

        human_pose.kpt17.x = self.keypoints_3d[id][17][0]
        human_pose.kpt17.y = self.keypoints_3d[id][17][1]
        human_pose.kpt17.z = self.keypoints_3d[id][17][2]

        return human_pose

    def publishOnRos(self, id = 0, TOPICLOGFLAG = False):

        if rospy.is_shutdown():
            print("SHUTDOWN")
            #TODO: flag de erro
            return 55
        
        human_pose = Human_Pose()

        human_pose = self.setMessage(human_pose)

        if(TOPICLOGFLAG):
            rospy.loginfo("I publish:")
            rospy.loginfo(human_pose)

        self.rosPub.publish(human_pose)
        self.rate.sleep()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=640x480, Recommends : 320x272 or 432x368 or 640x480 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    rspose = RsHumanPose(args)

    fps_time = 0
    while True:
        rspose.updatePose()
        rspose.printValues()
        cv2.putText(rspose.color_image_BGR,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        rspose.drawAndDisplay()
        rspose.plotPose3D()
        rspose.publishOnRos()
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()