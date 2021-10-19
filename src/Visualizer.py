import time
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import sys

class Visualizer:
    
    def __init__(self):
    
        # Define edges
        self.pairs_COCO = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]
        
        self.pairs_MPII = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (7, 6), (8, 7), (9, 8), (7, 12), (12, 11), (11, 10), (7, 13), (13, 14), (14, 15)]
        
        self.pairs_OpenPoseMPII = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (1,14), (14,8), (8,9), (9,10), (14,11), (11,12), (12,13)]
        
        self.pairs_OpenPoseCOCO = [(1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(0,14),(0,15),(14,16),(15,17)]
        
        self.pairs_upperbody_MPII = [(2, 6), (6, 3), (7, 6), (8, 7), (9, 8), (7, 12), (12, 11), (11, 10), (7, 13), (13, 14), (14, 15)]
        
        self.pos_upperbody = [2,3,6,7,8,9,10,11,12,13,14,15]
        
        self.pairs_full_3D_nose = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
                           (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
        
        self.pairs_upperbody_nose = [(0,1),(0,4),(0,7),(7,8),(8,9),
                           (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
        
        self.pos_upperbody_nose = [0,1,4,7,8,9,10,11,12,13,14,15,16]
        
        # Without Nose on 3D pose
        self.pairs_full_3D = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
                           (8,10),(10,11),(11,12),(8,13),(13,14),(14,15)]
        
        self.pairs_upperbody = [(0,1),(0,4),(0,7),(7,8),(8,9),
                           (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
        
        self.pos_upperbody_no_nose = [0,1,4,7,8,9,10,11,12,13,14,15,16]
        
        self.frame_tick = 0

    def initWindows(self, RGB = True, Disparity = False):
        
        # Create windows
        self.win_name_rgb = 'RGB'
        self.win_name_depth = 'Disparity'
        plt.ion()
        self.fig = plt.figure()
        mng = plt.get_current_fig_manager()
        
        # Check if OS is win or not
        is_windows = sys.platform.startswith('win')
        
        # if is_windows:
        #     mng.window.state('zoomed')
        # else:
        #     mng.resize(*mng.window.maxsize())
        self.ax2D = self.fig.add_subplot(1, 3, 1)
        self.axD = self.fig.add_subplot(1, 3, 2)
        
        # cv2.namedWindow(self.win_name_rgb, cv2.WINDOW_AUTOSIZE)
        # cv2.moveWindow(self.win_name_rgb, 0, 0)
        
        # if Disparity:
        #     cv2.namedWindow(self.win_name_depth, cv2.WINDOW_AUTOSIZE)
        #     cv2.moveWindow(self.win_name_depth, 640, 0)

#TODO: rename to loadImage()
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

#TODO: rename to getFrame(cam)
    def getWebcamFrame(self):

        self.frame_tick = time.time()

        ret, frame = self.cam.read()

        if not ret:
            print("Error trying to get frame.")
            return []

        self.frame = frame

        return self.frame

    def drawSkeleton(self, image, keypoints2D_full, mode = "MPII", upper_body = False):
        
        if mode == "MPII":
            if upper_body:
                pairs = self.pairs_upperbody_MPII
            else:
                pairs = self.pairs_MPII
        
        if upper_body:
            for keypoint in range(len(keypoints2D_full)):
                if(keypoint in self.pos_upperbody):
                    x = int(keypoints2D_full[keypoint][0])
                    y = int(keypoints2D_full[keypoint][1])
                    cv2.circle(image, (x, y), 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                    #cv2.putText(image, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        
        else:
            for i in range(len(keypoints2D_full)):
                x = int(keypoints2D_full[i][0])
                y = int(keypoints2D_full[i][1])
                cv2.circle(image, (x, y), 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
                #cv2.putText(image, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        
        for pair in pairs:
            part1 = pair[0]
            part2 = pair[1]
            point1 = tuple(keypoints2D_full[part1].astype(int))
            point2 = tuple(keypoints2D_full[part2].astype(int))
            color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
            cv2.line(image, point1, point2, color, 4)
            # cv2.circle(image, point1, 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            # cv2.circle(image, point2, 2, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
            
        return image
    
    def drawPersonAxis(self, image, keypoints, imgpts):
        corner = tuple(keypoints[0].ravel())
        print(corner)
        img = cv2.line(image, corner, tuple(np.array(imgpts[0].ravel(), np.uint)), (255,0,0), 5)
        img = cv2.line(image, corner, tuple(np.array(imgpts[1].ravel(), np.uint)), (0,255,0), 5)
        img = cv2.line(image, corner, tuple(np.array(imgpts[2].ravel(), np.uint)), (0,0,255), 5)
        return img
    
    def showImage(self, frame = [], with_FPS = True, block = False, Disparity = False):
        
        if block is False:
            self.ax3D.clear()
        
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax2D.imshow(frame_rgb)
        
        if frame is not None:
            self.frame = frame

    def showDepthImage(self, depth = [], with_FPS = True, block = False, Disparity = False):
        
        self.axD.imshow(depth)
        
        # if not block:
        #     plt.pause(1)
        #     quit()

        # if block:
        #     plt.pause(-1)
        # if not Disparity:
        #     self.inference_time = time.time() - self.frame_tick
        #     cv2.putText(self.frame, str(round(1/self.inference_time,2)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
        # #     frame = cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)))
        #     cv2.imshow(self.win_name_rgb,frame)

        # if Disparity:
        #     cv2.imshow(self.win_name_depth, frame)
            
        # if not block:
        #     if cv2.waitKey(1) == 27:
        #         quit()

        # if block:
        #     cv2.waitKey(10000)  

    def initPlot3D(self):
        
        # 3D plot init 
        # plt.ion()
        if self.fig in locals():
            self.fig = plt.figure()
        
        self.ax3D = self.fig.add_subplot(1, 3, 3, projection = '3d')
        self.ax3D.azim = 0 
        self.ax3D.dist = 10
        self.ax3D.elev = 10

    def plotRealSenseDeproj(self, keypoints3D, upper_body = False, block = False, nose = False):
        
        pairs = self.pairs_upperbody_MPII
        pos = self.pos_upperbody
        
        RADIUS = 1 # space around the subject
        self.ax3D.set_xlim3d([-RADIUS, RADIUS])
        self.ax3D.set_zlim3d([-RADIUS, RADIUS])
        self.ax3D.set_ylim3d([-RADIUS, RADIUS])
        
        x, y, z = [], [], []
        if upper_body:
            for keypoint in range(len(keypoints3D)):
                if(keypoints3D[keypoint][0] != -1) and (keypoint in pos):
                    x.append(keypoints3D[keypoint][0])
                    y.append(keypoints3D[keypoint][1])
                    z.append(keypoints3D[keypoint][2])
        else:
            for keypoint in range(len(keypoints3D)):
                if(keypoints3D[keypoint][0] != -1):
                    x.append(keypoints3D[keypoint][0])
                    y.append(keypoints3D[keypoint][1])
                    z.append(keypoints3D[keypoint][2])
            
        for edges in pairs:
                
            if(keypoints3D[edges[0]] == [-1, -1] or keypoints3D[edges[1]] == [-1, -1]):
                continue
            
            x1 = keypoints3D[edges[0]][0]
            y1 = keypoints3D[edges[0]][1]
            z1 = keypoints3D[edges[0]][2]

            x2 = keypoints3D[edges[1]][0]
            y2 = keypoints3D[edges[1]][1]
            z2 = keypoints3D[edges[1]][2]
            
            self.ax3D.plot([-x1, -x2],[-z1, -z2] ,[-y1, -y2])
        
        # Draw keypoints
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        self.ax3D.scatter(-x, -z, -y)
        
        plt.show(block=block)
        
        if not block:
            plt.pause(1)

        if block:
            plt.pause(-1)
        
        
    def drawBones3D(self, ax3D, keypoints3D, pairs, upper_body):
        
        x, y, z = [], [], []
        if upper_body:
            for keypoint in range(len(keypoints3D)):
                if(keypoints3D[keypoint][0] != -1) and (keypoint in self.pos_upperbody):
                    x.append(keypoints3D[keypoint][0])
                    y.append(keypoints3D[keypoint][1])
                    z.append(keypoints3D[keypoint][2])
        else:
            for keypoint in range(len(keypoints3D)):
                if(keypoints3D[keypoint][0] != -1):
                    x.append(keypoints3D[keypoint][0])
                    y.append(keypoints3D[keypoint][1])
                    z.append(keypoints3D[keypoint][2])
            
        for edges in pairs:
                
            if(keypoints3D[edges[0]] == [-1, -1] or keypoints3D[edges[1]] == [-1, -1]):
                continue
            
            x1 = keypoints3D[edges[0]][0]
            y1 = keypoints3D[edges[0]][1]
            z1 = keypoints3D[edges[0]][2]

            x2 = keypoints3D[edges[1]][0]
            y2 = keypoints3D[edges[1]][1]
            z2 = keypoints3D[edges[1]][2]
            
            # ax3D.plot([-x1, -x2],[-z1, -z2] ,[-y1, -y2])
            ax3D.plot([x1, x2],[y1, y2], [z1, z2])

        # Draw keypoints
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        # ax3D.scatter(-x, -z, -y)
        ax3D.scatter(x, y, z)
        
        return ax3D
    
    def onlyGroundAx(self, ax3D):
        
        # Get rid of the ticks and tick labels
        ax3D.set_xticks([])
        ax3D.set_yticks([])
        ax3D.set_zticks([])

        ax3D.get_xaxis().set_ticklabels([])
        ax3D.get_yaxis().set_ticklabels([])
        ax3D.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        ax3D.w_xaxis.set_pane_color(white)
        ax3D.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        ax3D.w_xaxis.line.set_color(white)
        ax3D.w_yaxis.line.set_color(white)
        ax3D.w_zaxis.line.set_color(white)
        
        return ax3D
        
    def plotPose3D(self, keypoints3D, upper_body = False, block = False, nose = False):
        
        # Clear buff
        # self.ax.clear()
        if nose:
            pairs = self.pairs_upperbody_nose
            pos = self.pos_upperbody_nose
        elif upper_body:
            pairs = self.pairs_full_3D
            pos = self.pos_upperbody
        else:
            pairs = self.pairs_full_3D
        
        RADIUS = 2000 # space around the subject
        self.ax3D.set_xlim3d([-RADIUS, RADIUS])
        self.ax3D.set_zlim3d([-RADIUS, RADIUS])
        self.ax3D.set_ylim3d([-RADIUS ,RADIUS])
        
        self.ax3D.set_xlabel('x')
        self.ax3D.set_ylabel('y')
        self.ax3D.set_zlabel('z')
        
        # Plot keypoints 3D
        # keypoints3D = keypoints3D/1000
        self.ax3D = self.drawBones3D(self.ax3D, keypoints3D, pairs, upper_body)
        
        # Make the enviroment with only one plan
        # self.ax3D = self.onlyGroundAx(self.ax3D)
        
        # Display 3D plot
        plt.show(block=block)
        
        if not block:
            plt.pause(1)

        if block:
            plt.pause(-1)
        
    #TODO
    def plot3DHUMAN36(self, keypoints3D, outputs, block = False):
        
        
        
        if block is False:
            self.ax3D.clear()
        
        self.ax3D.set_xlabel("X")
        self.ax3D.set_ylabel("Y")
        self.ax3D.set_zlabel("Z")
        
        RADIUS = 750 # space around the subject
        self.ax3D.set_xlim3d([-RADIUS, RADIUS])
        self.ax3D.set_zlim3d([-RADIUS, RADIUS])
        self.ax3D.set_ylim3d([-RADIUS, RADIUS])
        
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
            # self.ax.plot(-xline, -zline, -yline, lw=2, c=lcolor if LR[i] else rcolor)
            # self.ax.plot(xline, yline, zline, lw=2, c=lcolor if LR[i] else rcolor)
            self.ax3D.plot(-xline, -zline, -yline, lw=3, c=lcolor if LR[i] else rcolor)
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
        self.ax3D.scatter(-x, -z, -y)
        # self.ax.scatter(x, y, z)
        
        # Get rid of the ticks and tick labels
        self.ax3D.set_xticks([])
        self.ax3D.set_yticks([])
        self.ax3D.set_zticks([])

        self.ax3D.get_xaxis().set_ticklabels([])
        self.ax3D.get_yaxis().set_ticklabels([])
        self.ax3D.set_zticklabels([])

        # Get rid of the panes (actually, make them white)
        white = (1.0, 1.0, 1.0, 0.0)
        self.ax3D.w_xaxis.set_pane_color(white)
        self.ax3D.w_yaxis.set_pane_color(white)
        # Keep z pane

        # Get rid of the lines in 3d
        self.ax3D.w_xaxis.line.set_color(white)
        self.ax3D.w_yaxis.line.set_color(white)
        self.ax3D.w_zaxis.line.set_color(white)
        
        plt.draw()
        plt.show()
        
        if block:
            plt.pause(-1)
        else:
            plt.pause(0.001)
            
    def plotFullScene(persons):
        
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection = '3d')
        
        