import torch
import torchvision
import torch.nn as nn
import os
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

class Linear(nn.Module):

    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class LinearModel(nn.Module):
    def __init__(self,
                 linear_size=1024,
                 num_stage=2,
                 p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size =  16 * 2
        # 3d joints
        self.output_size = 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        return y

class SeffPose:

    def defineModel(self):

        self.model = LinearModel()
        ckpt = torch.load(os.getcwd() + '../models/ckpt_best.pth.tar')
        # load statistics data
        stat_3d = torch.load(os.getcwd() + '../models/stat_3d.pth.tar')

        print(stat_3d.keys())
        print(ckpt.keys())

        self.model.load_state_dict(ckpt['state_dict'])
        
    def bridge2D(self, keypoints2D_output, output_format='MPI'):
        
        if(output_format=='MPI'):
            pelvis = keypoints2D_output[8]/2 + keypoints2D_output[11]/2
            self.keypoints2D_Baseline = np.array([keypoints2D_output[10], keypoints2D_output[9], keypoints2D_output[8],
                                                  keypoints2D_output[11], keypoints2D_output[12], keypoints2D_output[13],
                                                  pelvis, keypoints2D_output[15], keypoints2D_output[1],
                                                  keypoints2D_output[0], keypoints2D_output[4], keypoints2D_output[3],
                                                  keypoints2D_output[2], keypoints2D_output[5], keypoints2D_output[6],
                                                  keypoints2D_output[15]])

    def estimatePose3Dfrom2DKeypoints(self, keypoints2D):
    
        outputs = self.model(keypoints2d)
        return outputs
