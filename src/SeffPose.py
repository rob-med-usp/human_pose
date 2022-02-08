from numpy.core.arrayprint import _make_options_dict
from numpy.lib.utils import _split_line
from numpy.testing._private.nosetester import NoseTester
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
    def __init__(self):
        
        self.use_cuda = torch.cuda.is_available()
        
                # Full mean per joints for all 17 that HUMAN3.6M has
        self.full_gtposeHM36m2D_mean = np.array([
                [532.08406529, 419.70004286],
                [532.29762931, 421.26396034],
                [531.93837415, 494.71912543],
                [529.71838862, 578.96493594],
                [531.80992425, 418.23858125],
                [530.68476821, 493.53829669],
                [529.3695928 , 575.96907295],
                [532.9368693 , 370.6273594 ],
                [534.11021786, 317.87395159],
                [534.55449896, 304.20719197],
                [534.86981418, 282.27538728],
                [534.11269724, 330.08322252],
                [533.53517702, 376.24164594],
                [533.49227803, 391.70207981],
                [533.52619558, 330.06556364],
                [532.50905212, 374.16059134],
                [532.72893539, 380.60538419]])
        
        # Full standard deviation per joints for all 17 that HUMAN3.6M has
        self.full_gtposeHM36m2D_std =  np.array(
                [[107.71979663, 63.35148791],
                [101.94373457, 62.88696832],
                [106.23227544, 48.41074155],
                [108.45089997, 54.58495702],
                [118.99125297, 64.11214135],
                [119.10602351, 50.53679963],
                [120.5990632 , 56.38689222],
                [109.0578714 , 68.6936446 ],
                [111.18250122, 74.86112372],
                [111.61261415, 77.79330958],
                [113.20325395, 79.89477859],
                [105.69981542, 73.2588583 ],
                [107.05346824, 73.92571615],
                [107.98747907, 83.29661543],
                [121.58758131, 74.24605525],
                [134.32906881, 77.47220667],
                [131.80190209,  89.8535445]])
        
    def HUMAN36MtoMPIIstats(self, keypoints2D_HM36m):
        
        hip        = keypoints2D_HM36m[0]
        r_hip      = keypoints2D_HM36m[1]
        r_knee     = keypoints2D_HM36m[2]
        r_ankle    = keypoints2D_HM36m[3]
        l_hip      = keypoints2D_HM36m[4]
        l_knee     = keypoints2D_HM36m[5]
        l_ankle    = keypoints2D_HM36m[6]
        spine      = keypoints2D_HM36m[7]
        mid_thorax = keypoints2D_HM36m[8]
        nose       = keypoints2D_HM36m[9] 
        head_top   = keypoints2D_HM36m[10]
        l_shoulder = keypoints2D_HM36m[11] 
        l_elbow    = keypoints2D_HM36m[12]
        l_wrist    = keypoints2D_HM36m[13]
        r_shoulder = keypoints2D_HM36m[14]
        r_elbow    = keypoints2D_HM36m[15]
        r_wrist    = keypoints2D_HM36m[16]
        
        # Init MPII kpts
        keypoints2D_MPII = np.zeros((16,2))
        
        # Calculate pelvis
        pelvis = l_hip + r_hip
        pelvis = pelvis/2
        
        #calculate upper_neck
        upper_neck = nose
        
        # Set array
        keypoints2D_MPII[0] = r_ankle
        keypoints2D_MPII[1] = r_knee
        keypoints2D_MPII[2] = r_hip
        keypoints2D_MPII[3] = l_hip
        keypoints2D_MPII[4] = l_knee
        keypoints2D_MPII[5] = l_ankle
        #test1: spine->pelvis
        keypoints2D_MPII[6] = spine
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

    def defineModel(self, net = "GT"):
        # Set model arch
        self.model = LinearModel()
        
        # Choose pre-trained model: 'SH' or 'GT'
        self.net = net
        
        # set the computation device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Hydrapose path
        path = os.path.normpath(os.path.dirname(__file__)+os.sep+os.pardir)
        if net == 'GT':
            # Remove Neck/Nose from stat 2D
            self.stat2D_mean = np.delete(self.full_gtposeHM36m2D_mean, 9, 0)
            self.stat2D_std = np.delete(self.full_gtposeHM36m2D_std, 9, 0)
            if not self.use_cuda:
                
                ckpt = torch.load(os.path.join(path, 'models', 'seffpose', 'human36_gt_iter200.pth.tar'),  map_location=torch.device('cpu'))
            else:
                ckpt = torch.load(os.path.join(path, 'models', 'seffpose', 'human36_gt_iter200.pth.tar'))

        elif net == 'SH':
            # Adapt 2D stats for MPII entry
            self.stat2D_mean = self.HUMAN36MtoMPIIstats(self.full_gtposeHM36m2D_mean)
            self.stat2D_std = self.HUMAN36MtoMPIIstats(self.full_gtposeHM36m2D_std)
            if not self.use_cuda:
                ckpt = torch.load(os.path.join(path,'models','seffpose','hm36m_sh_iter138.pth.tar'), map_location=torch.device('cpu'))
            else:
                ckpt = torch.load(os.path.join(path,'models','seffpose','hm36m_sh_iter138.pth.tar'))
                
        # Load statistics data
        self.stat_3d = torch.load(os.path.join(path,'models','seffpose','stat_3d.pth.tar'))
        
        # Load the modle on to the computation device and set to eval mode
        self.model.to(self.device).eval()
        # print(stat_3d.keys())
        # print(ckpt.keys())

        self.model.load_state_dict(ckpt['state_dict'])
    
    def normalizePose2D(self, keypoints2D_HM36M, Width, Heigth):
        
        # # divide by image width and length (width = length = 1000 pixels)
        # gtpose2D_mean = gtpose2D_mean/1000
        # gtpose2D_std = gtpose2D_std/1000
        
        # # Adapt mean and std to image scale
        # adapted_mean = self.gtpose2DMPII_mean * square_image_length / 1000
        # adapted_std = self.gtpose2DMPII_std * square_image_length / 1000 
        
        # # Normalize keypoints2D pre-normalized
        # keypoints2D_norm = keypoints2D - adapted_mean
        # keypoints2D_norm = keypoints2D_norm / adapted_std
        
        # Make a copy to avoid interference
        keypoints2D = keypoints2D_HM36M.copy()
        
        # Doesnt work
        # # Simulate a 1000 pixels image
        # keypoints2D[:,0] = 500 - Width/2 + keypoints2D[:, 0] 
        # keypoints2D[:,1] = 500 - Heigth/2 + keypoints2D[:, 1]
        
        # Prepare
        keypoints2D[:, 0] = keypoints2D[:, 0] / Width 
        keypoints2D[:, 1] = keypoints2D[:, 1] / Heigth 
        # Normalize
        keypoints2D_norm = keypoints2D - self.stat2D_mean/1000
        keypoints2D_norm = keypoints2D_norm / self.stat2D_std * 1000
        
        return keypoints2D_norm
        
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
    
    def estimatePose3Dfrom2DKeypoints(self, keypoints2D):
        #keypoints2D = self.COCOtoMPII(keypoints2D)
        #self.keypoints2D = keypoints2D
        
        keypoints2D = np.reshape(keypoints2D, (1,32))
        
        keypoints2D = keypoints2D.astype(np.float32)
        
        keypoints2D = torch.from_numpy(keypoints2D)
        torch.no_grad()
        outputs = self.model(keypoints2D)
        
        self.outputs = self.unNormalizeData(outputs.data.cpu().numpy(), self.stat_3d['mean'], self.stat_3d['std'], self.stat_3d['dim_use'])
        
        # remove dim ignored
        dim_use = np.hstack((np.arange(3), self.stat_3d['dim_use']))
        
        keypoints3D = self.outputs[:, dim_use]
        keypoints3D = np.reshape(keypoints3D[0], (17,3))
        
        return keypoints3D

    def getMPIIKeypoints(self):
        return self.keypoints2D
