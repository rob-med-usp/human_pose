from typing import KeysView
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
        ckpt = torch.load(os.getcwd() + '/models/ckpt_best.pth.tar')
        # load statistics data
        self.stat_3d = torch.load(os.getcwd() + '/models/stat_3d.pth.tar')

        self.model.eval()
        # print(stat_3d.keys())
        # print(ckpt.keys())

        self.model.load_state_dict(ckpt['state_dict'])
    
    # TODO
    def OpenPoseCOCOtoCOCO(self, keypoints2D_output, output_format='MPII'):
        
        # Stacked Hourglass produces 16 joints. These are the names. Ref:https://github.com/una-dinosauria/3d-pose-baseline/blob/666080d86a96666d499300719053cc8af7ef51c8/src/data_utils.py#L16
        # Seff uses MPI keypoints w/ 16 joints in this order
        # [0]'RFoot'
        # [1]'RKnee'
        # [2]'RHip'
        # [3]'LHip'
        # [4]'LKnee'
        # [5]'LFoot'
        # [6]'Hip'
        # [7]'Spine'
        # [8]'Thorax'
        # [9]'Head'
        # [10]'RWrist'
        # [11]'RElbow'
        # [12]'RShoulder'
        # [13]'LShoulder'
        # [14]'LElbow'
        # [15]'LWrist'
        
        nose = keypoints2D_OpenPoseCOCO[0]
        upper_thorax = keypoints2D_OpenPoseCOCO[1]
        r_shoulder = keypoints2D_OpenPoseCOCO[2]
        r_elbow = keypoints2D_OpenPoseCOCO[3]
        r_wrist = keypoints2D_OpenPoseCOCO[4]
        l_shoulder = keypoints2D_OpenPoseCOCO[5]
        l_elbow = keypoints2D_OpenPoseCOCO[6]
        l_wrist = keypoints2D_OpenPoseCOCO[7]
        r_hip = keypoints2D_OpenPoseCOCO[8]
        r_knee = keypoints2D_OpenPoseCOCO[9]
        r_ankle = keypoints2D_OpenPoseCOCO[10]
        l_hip = keypoints2D_OpenPoseCOCO[11]
        l_knee = keypoints2D_OpenPoseCOCO[12]
        l_ankle = keypoints2D_OpenPoseCOCO[13]
        r_eye = keypoints2D_OpenPoseCOCO[14]
        l_eye = keypoints2D_OpenPoseCOCO[15]
        r_ear = keypoints2D_OpenPoseCOCO[16]
        l_ear = keypoints2D_OpenPoseCOCO[17]
        bkg = keypoints2D_OpenPoseCOCO[18]
           
    # TODO
    def OpenPoseMPIItoMPII(self):
        if(output_format=='MPII'):
            pelvis = keypoints2D_output[8]/2 + keypoints2D_output[11]/2
            self.keypoints2D_Baseline = np.array([keypoints2D_output[10], keypoints2D_output[9], keypoints2D_output[8],
                                                  keypoints2D_output[11], keypoints2D_output[12], keypoints2D_output[13],
                                                  pelvis, keypoints2D_output[15], keypoints2D_output[1],
                                                  keypoints2D_output[0], keypoints2D_output[4], keypoints2D_output[3],
                                                  keypoints2D_output[2], keypoints2D_output[5], keypoints2D_output[6],
                                                  keypoints2D_output[15]])
            
            self.keypoints2D_Baseline = np.reshape(self.keypoints2D_Baseline, (1,32))
            
            self.keypoints2D_Baseline = self.keypoints2D_Baseline.astype(np.float32)
            
            self.keypoints2D_Baseline = torch.from_numpy(self.keypoints2D_Baseline)

    def unNormalizeData(self, normalized_data, data_mean, data_std, dimensions_to_use):
        T = normalized_data.shape[0]  # Batch size
        D = data_mean.shape[0]  # 96
        
        orig_data = np.zeros((T, D), dtype=np.float32)

        orig_data[:, dimensions_to_use] = normalized_data

        # Multiply times stdev and add the mean
        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        orig_data = np.multiply(orig_data, stdMat) + meanMat
        return orig_data
    
    def COCOtoMPII(self, keypoints2D_COCO):
    
        # COCO:
        #https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
        # "nose",        "left_eye",      "right_eye",      "left_ear",
        # "right_ear",   "left_shoulder", "right_shoulder", "left_elbow",
        # "right_elbow", "left_wrist",    "right_wrist",    "left_hip", 
        # "right_hip",   "left_knee",     "right_knee",     "left_ankle",
        # "right_ankle"

        # Set auxiliar variables
        nose = keypoints2D_COCO[0]
        l_eye = keypoints2D_COCO[1]
        r_eye = keypoints2D_COCO[2]
        l_ear = keypoints2D_COCO[3]
        r_ear = keypoints2D_COCO[4]
        l_shoulder = keypoints2D_COCO[5]
        r_shoulder = keypoints2D_COCO[6]
        l_elbow = keypoints2D_COCO[7]
        r_elbow = keypoints2D_COCO[8]
        l_wrist = keypoints2D_COCO[9]
        r_wrist = keypoints2D_COCO[10]
        l_hip = keypoints2D_COCO[11]
        r_hip = keypoints2D_COCO[12]
        l_knee = keypoints2D_COCO[13]
        r_knee = keypoints2D_COCO[14]
        l_ankle = keypoints2D_COCO[15]
        r_ankle = keypoints2D_COCO[16]
        
        # MPII:
        #http://human-pose.mpi-inf.mpg.de/#download
        #  r ankle,    r knee,     r hip,   l hip,
        #  l knee,     l ankle,    pelvis,  thorax,
        #  upper neck, head top,   r wrist, r elbow,
        #  r shoulder, l shoulder, l elbow, l wrist
        
        # Calculate pelvis
        pelvis = l_hip + r_hip
        pelvis = pelvis/2
        
        # Calculate upper_neck
        upper_neck = r_shoulder + l_shoulder
        upper_neck = upper_neck/2
        
        # Calculate thorax
        mid_thorax = pelvis + upper_neck
        mid_thorax = mid_thorax/2
        
        # Calculate head_top
        head_center = nose + r_ear + l_ear + r_eye + l_eye
        head_center = head_center/5
        vector_head = head_center - upper_neck
        head_top = upper_neck + 2*vector_head
        
        # Init MPII kpts
        keypoints2D_MPII = np.zeros((16,2))
        print(r_ankle)
        # Set array
        keypoints2D_MPII[0] = r_ankle
        keypoints2D_MPII[1] = r_knee
        keypoints2D_MPII[2] = r_hip
        keypoints2D_MPII[3] = l_hip
        keypoints2D_MPII[4] = l_knee
        keypoints2D_MPII[5] = l_ankle
        keypoints2D_MPII[6] = pelvis
        keypoints2D_MPII[7] = mid_thorax
        keypoints2D_MPII[8] = upper_neck
        keypoints2D_MPII[9] = head_top
        keypoints2D_MPII[10] = r_wrist
        keypoints2D_MPII[11] = r_elbow
        keypoints2D_MPII[12] = r_shoulder
        keypoints2D_MPII[13] = l_shoulder
        keypoints2D_MPII[14] = l_elbow
        keypoints2D_MPII[15] = l_wrist
        
        return keypoints2D_MPII
        
    def estimatePose3Dfrom2DKeypoints(self, keypoints2D):
        
        keypoints2D = self.COCOtoMPII(keypoints2D)
    
        keypoints2D = np.reshape(keypoints2D, (1,32))
        
        keypoints2D = keypoints2D.astype(np.float32)
        
        keypoints2D = torch.from_numpy(keypoints2D)
        
        outputs = self.model(keypoints2D)
        
        self.outputs = self.unNormalizeData(outputs.data.cpu().numpy(), self.stat_3d['mean'], self.stat_3d['std'], self.stat_3d['dim_use'])
        
        # remove dim ignored
        dim_use = np.hstack((np.arange(3), self.stat_3d['dim_use']))
        
        keypoints3D = self.outputs[:, dim_use]
        keypoints3D = np.reshape(keypoints3D[0], (17,3))
        
        return keypoints3D

    def getRawOutputs(self):
        return self.outputs
    
# def convert_coco_to_openpose_cords(coco_keypoints_list):
#     # coco keypoints: [x1,y1,v1,...,xk,yk,vk]       (k=17)
#     #     ['Nose', Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
#     #      'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank']
#     # openpose keypoints: [y1,...,yk], [x1,...xk]   (k=18, with Neck)
#     #     ['Nose', *'Neck'*, 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri','Rhip',
#     #      'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
#     indices = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]
#     y_cords = []
#     x_cords = []
#     for i in indices:
#         xi, yi, vi = coco_keypoints_list[i*3:(i+1)*3]
#         if vi == 0: # not labeled
#             y_cords.append(MISSING_VALUE)
#             x_cords.append(MISSING_VALUE)
#         elif vi == 1:   # labeled but not visible
#             y_cords.append(yi)
#             x_cords.append(xi)
#         elif vi == 2:   # labeled and visible
#             y_cords.append(yi)
#             x_cords.append(xi)
#         else:
#             raise ValueError("vi value: {}".format(vi))
#     # Get 'Neck' keypoint by interpolating between 'Lsho' and 'Rsho' keypoints
#     l_shoulder_index = 5
#     r_shoulder_index = 6
#     l_shoulder_keypoint = coco_keypoints_list[l_shoulder_index*3:(l_shoulder_index+1)*3]
#     r_shoulder_keypoint = coco_keypoints_list[r_shoulder_index*3:(r_shoulder_index+1)*3]
#     if l_shoulder_keypoint[2] > 0 and r_shoulder_keypoint[2] > 0:
#         neck_keypoint_y = int((l_shoulder_keypoint[1]+r_shoulder_keypoint[1])/2.)
#         neck_keypoint_x = int((l_shoulder_keypoint[0]+r_shoulder_keypoint[0])/2.)
#     else:
#         neck_keypoint_y = neck_keypoint_x = MISSING_VALUE
#     open_pose_neck_index = 1
#     y_cords.insert(open_pose_neck_index, neck_keypoint_y)
#     x_cords.insert(open_pose_neck_index, neck_keypoint_x)
#     return np.concatenate([np.expand_dims(y_cords, -1),
#                         np.expand_dims(x_cords, -1)], axis=1)