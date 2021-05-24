import time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

class Visualizer:
    
    def __init__(self):
    
        # Define edges
        self.pairs_COCO = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.pairs_MPII = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (7, 6), (8, 7), (9, 8), (8, 12), (12, 11), (11, 10), (8, 13), (13, 14), (14, 15)]
        
        self.pairs_OpenPoseMPII = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,14), (14,8), (8,9), (9,10), (14,11), (11,12), (12,13)]
        
        self.pairs_OpenPoseCOCO = [(1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(0,14),(0,15),(14,16),(15,17)]
                
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

    def drawSkeleton(self, image, keypoints2D, mode = "MPII"):
        
        if mode == "MPII":
            pairs = self.pairs_MPII
        
        for i in range(len(keypoints2D)):
            x = int(keypoints2D[i][0])
            y = int(keypoints2D[i][1])
            cv2.circle(image, (x, y), 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(image, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        
        for pair in pairs:
            part1 = pair[0]
            part2 = pair[1]
            point1 = tuple(keypoints2D[part1].astype(int))
            point2 = tuple(keypoints2D[part2].astype(int))
            cv2.line(image, point1, point2, (0, 255, 255), 2)
            cv2.circle(image, point1, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(image, point2, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
        return image
    
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
        
        # self.ax.set_xlim(-1, 1)
        # self.ax.set_ylim(0, 4)
        # self.ax.set_zlim(-2, 2)

    def plotPose3DbyID(self, id, keypoints3D, pairs_mode = 'MPI'):
    
        if(pairs_mode == 'MPI'):
            if(len(keypoints3D) == 17):
                self.keypoints3D = keypoints3D
        
        if(pairs_mode == 'COCO'):
            if(len(keypoints3D) == 17):
                self.keypoints3D = keypoints3D
        
        # Clear buff
        self.ax.clear()
        
        # TODO: eliminate this for
        x, y, z = [], [], []
        for keypoint in range(len(self.keypoints3D)):
            if(self.keypoints3D[keypoint][0] != -1):
                x.append(self.keypoints3D[keypoint][0])
                y.append(self.keypoints3D[keypoint][1])
                z.append(self.keypoints3D[keypoint][2])
        
        # Draw bones
        if(self.pose2D_mode == "OpenPose"):
            pairs = self.pairs_OpenPose
        if(self.pose2D_mode == "RCNN"):
            pairs = self.pairs_RCNN
            
        for edges in pairs:
                
            if(self.keypoints3D[edges[0]] == [-1, -1] or self.keypoints3D[edges[1]] == [-1, -1]):
                continue
            
            x1 = self.keypoints3D[edges[0]][0]
            y1 = self.keypoints3D[edges[0]][1]
            z1 = self.keypoints3D[edges[0]][2]

            x2 = self.keypoints3D[edges[1]][0]
            y2 = self.keypoints3D[edges[1]][1]
            z2 = self.keypoints3D[edges[1]][2]
            
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
        
    #TODO
    def plot3DHUMAN36(self, keypoints3D, outputs):

        x, y, z = [], [], []
        for keypoint in range(len(keypoints3D)):
            x.append(keypoints3D[keypoint][0])
            y.append(keypoints3D[keypoint][1])
            z.append(keypoints3D[keypoint][2])

        I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
        J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
        LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        lcolor="#3498db"
        rcolor="#e74c3c"

        outputs = np.reshape(outputs, (32, -1))
        # Make connection matrix
        for i in np.arange( len(I) ):
            xline, yline, zline = [np.array( [outputs[I[i], j], outputs[J[i], j]] ) for j in range(3)]
            self.ax.plot(xline, -zline, -yline, lw=2, c=lcolor if LR[i] else rcolor)

        # for i in :

        #     x1 = keypoints3D[I[]][0]
        #     y1 = keypoints3D[][1]
        #     z1 = keypoints3D[][2]

        #     x2 = keypoints3D[][0]
        #     y2 = keypoints3D[][1]
        #     z2 = keypoints3D[][2]

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        #ax.scatter(x, z, -y)
        self.ax.scatter(x, -z, -y)
        plt.draw()
        plt.show()
        plt.pause(-1)