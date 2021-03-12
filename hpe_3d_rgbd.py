import argparse
import time
import os

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

import pyrealsense2 as rs 

# Debug flag
DEBUG = True

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    
    # Model params
    w, h = model_wh(args.resize)
    
    # Model construction
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    if DEBUG:
        print("Device: " + device_product_line + '\n')

    # Enable stream for RealSense
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    pc = rs.pointcloud()

    colorizer = rs.colorizer()

    # initialize fps
    fps_time = 0
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
             
            if not depth_frame or not color_frame:
                continue
            
            #print(color_frame.get_data())

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image_BGR = np.asanyarray(color_frame.get_data())

            mapped_frame = color_frame
            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)
            
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics 
        

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image_BGR.shape

            # OpenPose TF1.5.1 (TODO: keras implementation)
            humans = e.inference(color_image_BGR, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            
            # initialize 2d keypoints vector
            keypoints_2d_norm = np.array(np.zeros((10,18,2)))
            keypoints_2d = np.array(np.zeros((10,18,2)))
            humans_iter = 0

            for human in humans:
                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                        continue

                    body_part = human.body_parts[i]
                    
                    keypoints_2d_norm[humans_iter][i][0] = body_part.x 
                    keypoints_2d_norm[humans_iter][i][1] = body_part.y
                humans_iter += 1

            for id in range(len(keypoints_2d_norm)):
                for body in range(len(keypoints_2d_norm[id])):
                    if (keypoints_2d_norm[id][body][0] != 0): 
                        keypoints_2d[id][body][0] = keypoints_2d_norm[id][body][0]*color_image_BGR.shape[1] 
                        keypoints_2d[id][body][1] = keypoints_2d_norm[id][body][1]*color_image_BGR.shape[0]

            keypoints_2d = np.where(keypoints_2d == 0., -1, keypoints_2d)

            keypoints_3d = np.array(np.zeros((10,18,3)))

            for id in range(len(keypoints_2d)):
                for body in range(len(keypoints_2d[id])):
                    if(keypoints_2d[id][body][0] != -1 and keypoints_2d[id][body][1] != -1):
                        x = np.int(keypoints_2d[id][body][0])
                        y = np.int(keypoints_2d[id][body][1])
                        keypoints_3d[id][body] = rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth_frame.get_distance(x, y))
                    else:
                        keypoints_3d[id][body] = np.array([-1,-1,-1])

            if DEBUG:
                os.system('clear')
                print('| Printing 2D (u, v) in pixels and 3D (x, y, z) in meters keypoints values: ')
                for body in range(18):
                    print('| ' + str(common.CocoPart(body)) + ' = ' + '(' + str(keypoints_2d[0][body][0]) + ', ' + str(keypoints_2d[0][body][1]) + 
                    ')   (' + str(keypoints_3d[0][body][0]) + ', ' + str(keypoints_3d[0][body][1]) + ', ' + str(keypoints_3d[0][body][2]) +')')

            humans = np.asanyarray(humans)
            print("humans.shape (numbers of humans) = " + str(humans.shape))

            color_image_BGR = TfPoseEstimator.draw_humans(color_image_BGR, humans, imgcopy=False)
            #############################################
                    
            # show FPS on image
            cv2.putText(color_image_BGR,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # Display images
            cv2.imshow('color image BGR', color_image_BGR)
            cv2.imshow('depth colorized', depth_colormap)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break

    finally:
        #Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
