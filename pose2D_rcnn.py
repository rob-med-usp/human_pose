# 3D Pose Estimation with Keypoint-RCNN and RealSense stereo camera

import cv2
import matplotlib
import torch
import torchvision
import numpy as np
import argparse
import time

from PIL import Image
from torchvision.transforms import transforms as transforms

class Pose2D_RCNN:

    def __init__(self):
        
        self.edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]

        self.use_cuda = torch.cuda.is_available()

    def defineModel(self):

        # initialize the model
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                    num_keypoints=17)
        # set the computation device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        # load the modle on to the computation device and set to eval mode
        self.model.to(self.device).eval()
        # initialize transform obj
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def setWebcam(self, cam):

        self.cam = cv2.VideoCapture(cam)

        if not self.cam.isOpened():
            print("Error trying to open camera.")
            return False, []

    def getWebcamFrame(self):

        ret, frame = self.cam.read()

        if not ret:
            print("Error trying to get frame.")
            return False, []

        self.frame = frame

        return True, self.frame


    def predictFrame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # transform the image
        image = self.transform(rgb)

        # add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            self.outputs = self.model(image)
        self.inference_time = time.time() - t0
        
        return self.outputs

    def drawSkeleton(self, frame, outputs):

        # the `outputs` is list which in-turn contains the dictionaries
        for i in range(len(outputs[0]['keypoints'])):
            keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
            # proceed to draw the lines if the confidence score is above 0.9
            if outputs[0]['scores'][i] > 0.975:
                keypoints = keypoints[:, :].reshape(-1, 3)
                for p in range(keypoints.shape[0]):
                    # draw the keypoints
                    cv2.circle(frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                                3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    # uncomment the following lines if you want to put keypoint number
                    # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                for ie, e in enumerate(self.edges):
                    # get different colors for the edges
                    rgb = matplotlib.colors.hsv_to_rgb([
                        ie/float(len(self.edges)), 1.0, 1.0
                    ])
                    rgb = rgb*255
                    # join the keypoint pairs to draw the skeletal structure
                    cv2.line(frame, (keypoints[e, 0][0], keypoints[e, 1][0]),
                            (keypoints[e, 0][1], keypoints[e, 1][1]),
                            tuple(rgb), 2, lineType=cv2.LINE_AA)
            else:
                continue
        return frame

    def showImage(self, frame, with_FPS = True):

        cv2.putText(frame, str(round(1/self.inference_time,2)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0))
        cv2.imshow("Result",frame)

        if cv2.waitKey(1) == 27:
#            break
            return False

    def run():
        pass
