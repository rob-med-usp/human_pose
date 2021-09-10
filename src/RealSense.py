import pyrealsense2 as rs
import time
import os

import cv2
import numpy as np

class RealSense:
    
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

    def deprojectPose3D(self, keypoints2D):

        self.keypoints2D = keypoints2D

        self.keypoints3D = np.ones((len(keypoints2D), 3)) * (-1)

        for keypoint in range(len(keypoints2D)):
            if(keypoints2D[keypoint][0] != -1):
                x = np.int(keypoints2D[keypoint][0])
                y = np.int(keypoints2D[keypoint][1])
                self.keypoints3D[keypoint] = rs.rs2_deproject_pixel_to_point(self.color_intrin, [x,y], self.depth_frame.get_distance(x, y))
            else:
                self.keypoints3D[keypoint] = np.array([-1, -1, -1])

        return self.keypoints3D
    
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
        
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
    def getBagFrames(self):
        
        # Get frame set
        frames = self.pipeline_bag.wait_for_frames()
        
        # Align frames
        frames = self.align.process(frames)
        
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
