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

        use_cuda = torch.cuda.is_available()




