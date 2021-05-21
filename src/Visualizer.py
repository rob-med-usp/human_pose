import time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

class Visualizer:
    
    def __init__(self):
    
        # Define edges
        self.pairs_OpenPose = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.pairs_RCNN = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.frame_tick = 0

    def initWindows(self, RGB = True, Disparity = False):
        
        # Create windows
        self.win_name_rgb = 'RGB'
        self.win_name_depth = 'Disparity'
        
        cv2.namedWindow(self.win_name_rgb, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.win_name_rgb, 0, 0)
        
        if Disparity:
            cv2.namedWindow(self.win_name_depth, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.win_name_depth, 640, 0)

    def getImagefromFile(self, fname):
        
        image = cv2.imread(fname)
        
        if image is None:
            print("Image empty, Quiting...")
            quit()
            
        return image

    def setWebcam(self, cam):

        self.cam = cv2.VideoCapture(cam)

        if not self.cam.isOpened():
            print("Error trying to open camera.")
            return False, []

    def getWebcamFrame(self):

        self.frame_tick = time.time()

        ret, frame = self.cam.read()

        if not ret:
            print("Error trying to get frame.")
            return []

        self.frame = frame

        return self.frame

    def showImage(self, frame = [], with_FPS = True, block = False):

        if frame is not None:
            self.frame = frame

        self.inference_time = time.time() - self.frame_tick
        cv2.putText(self.frame, str(round(1/self.inference_time,2)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
        cv2.imshow(self.win_name_rgb,frame)

        if not block:
            if cv2.waitKey(1) == 27:
                quit()

        if block:
            cv2.waitKey()  

    def initPlot3D(self):
        
        # 3D plot init 
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection = '3d')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 4)
        self.ax.set_zlim(-2, 2)

    def plotPose3DbyID(self, id, keypoints3D, pairs_mode = 'MPI'):
        
        if(pairs_mode == 'MPI'):
            if(len(keypoints3D) == 16):
                self.keypoints3D = keypoints3D
        
        if(pairs_mode == 'COCO'):
            if(len(keypoints3D) == 17):
                self.keypoints3D = keypoints3D
        
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