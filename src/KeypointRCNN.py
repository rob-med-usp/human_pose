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

from SkeletonsBridge import *

class KeypointRCNN:

    def __init__(self):
        
        self.bridge = SkeletonsBridge()

        self.edges = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Starting device with CUDA...")
        else:
            print("Starting device with CPU...")
        

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
    
    def estimate2DPose(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # transform the image
        image = self.transform(rgb)

        # add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            self.outputs = self.model(image)
        self.inference_time = time.time() - t0
        
        persons, scores = self._getKeypointsAndPersonScores()
        
        persons = self._filterPersonsbyScore(persons, scores, min_thresh = 0.95)

        persons = self.bridge.transformPersonsFromTo(persons, "COCO", 'HM36M')

        return np.array(persons)
    
    def _filterPersonsbyScore(self, persons, scores, min_thresh = 0.4):
        persons_filtered = []
        for i in range(len(scores)):
            if scores[i] > min_thresh:
                persons_filtered.append(persons[i])
        
        return persons_filtered
        
    def _getKeypointsAndPersonScores(self):
        
        # Get keypoints
        self.persons = np.asanyarray(self.outputs[0]['keypoints'])
        
        # Get scores
        self.person_scores = np.asanyarray(self.outputs[0]['scores'])
        
        # Remove visibility collum
        self.persons = self.persons[:,:,:2]
        
        return self.persons, self.person_scores

    # # TODO: Recieve keypoints instead outputs
    # def drawSkeleton(self, frame, persons):

    #     # the `outputs` is list which in-turn contains the dictionaries
    #     for i in range(len(persons)):
            
    #         keypoints = persons[i]
    #         # proceed to draw the lines if the confidence score is above 0.9
    #         keypoints = keypoints[:, :].reshape(-1, 2)
    #         for p in range(keypoints.shape[0]):
    #             # draw the keypoints
    #             cv2.circle(frame, (int(keypoints[p, 0]), int(keypoints[p, 1])), 3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    #             # uncomment the following lines if you want to put keypoint number
    #             # cv2.putText(image, f"{p}", (int(keypoints[p, 0]+10), int(keypoints[p, 1]-5)),
    #             #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #         for ie, e in enumerate(self.edges):
    #             # get different colors for the edges
    #             rgb = matplotlib.colors.hsv_to_rgb([
    #                 ie/float(len(self.edges)), 1.0, 1.0
    #             ])
    #             rgb = rgb*255
    #             # join the keypoint pairs to draw the skeletal structure
    #             cv2.line(frame, (keypoints[e, 0][0], keypoints[e, 1][0]),
    #                     (keypoints[e, 0][1], keypoints[e, 1][1]),
    #                     tuple(rgb), 2, lineType=cv2.LINE_AA)
    #         else:
    #             continue
    #     return frame