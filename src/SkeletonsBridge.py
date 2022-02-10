import numpy as np

class SkeletonsBridge:
    
    def __init__(self):
        
        self.MVOR_kpts = ["nose","neck","lshould","rshould","lhip","rhip","lelb","relb","lwri","rwris"]
        
        self.pairs_MVOR = [[0, 1],[1, 3],[3, 7],[3, 5],[7, 9],[1, 2],[2, 4],[2, 6],[6, 8],[4, 5]]

        # COCO
        self.pairs_COCO = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
        (12, 14), (14, 16), (5, 6)]

        self.pairs_upper_COCO = [
        (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
        (5, 7), (7, 9), (5, 11), (6, 12), (5, 6)]

        # OpenPoseCOCO
        self.pairs_OpenPoseCOCO = [
        (1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),
        (8,9),(9,10),(1,11),(11,12),(12,13),(0,14),(0,15),(14,16),(15,17)]
        
        self.pairs_upper_OpenPoseCOCO = [
        (1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(1,11),(0,14),(0,15),(14,16),(15,17)]

        #MPII
        #TODO: see if 7 is thorax or chest
        self.pairs_MPII = [
        (0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(7,6),(8,7),(9,8),(7,12),(12,11),(11,10),(7,13),(13,14),(14,15)]
        
        self.pairs_upper_MPII = [
        (2,6),(6,3),(7,6),(8,7),(9,8),(7,12),(12,11),(11,10),(7,13),(13,14),(14,15)]

        #OpenPose MPII
        self.pairs_OpenPoseMPII = [
        (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,14),(14,8),(8,9),(9,10),(14,11),(11,12),(12,13)]
        
        self.pairs_upper_OpenPoseMPII = [
        (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,14),(14,8),(14,11)]

        #Human 3.6M 3D (with nose)
        self.pairs_Human36M_full = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
        (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]
        
        self.pairs_upper_Human3M_full = [(0,1),(0,4),(0,7),(7,8),(8,9),
        (9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]

        #Human 3.6M 2D (MPII compatible, without nose)
        self.pairs_Human36M_noseless = [
        (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),
        (8,10),(10,11),(11,12),(8,13),(13,14),(14,15)]
        
        self.pairs_upper_Human36M_noseless = [
        (0,1),(0,4),(0,7),(7,8),(8,9),
        (8,10),(10,11),(11,12),(8,13),(13,14),(14,15)]
        
        self.frame_tick = 0
    
    #MPII to HM36M 2D-> pose 2d input
    def MPIItoHM36M(self, keypoints2D_MPII):
        # MPII: http://human-pose.mpi-inf.mpg.de/#download
        #
        # Set array
        r_ankle      = keypoints2D_MPII[0] 
        r_knee       = keypoints2D_MPII[1] 
        r_hip        = keypoints2D_MPII[2] 
        l_hip        = keypoints2D_MPII[3] 
        l_knee       = keypoints2D_MPII[4] 
        l_ankle      = keypoints2D_MPII[5] 
        pelvis       = keypoints2D_MPII[6] 
        # upper_thorax = keypoints2D_MPII[7]
        thorax = keypoints2D_MPII[7] 
        upper_neck   = keypoints2D_MPII[8] 
        head_top     = keypoints2D_MPII[9] 
        r_wrist      = keypoints2D_MPII[10] 
        r_elbow      = keypoints2D_MPII[11] 
        r_shoulder   = keypoints2D_MPII[12] 
        l_shoulder   = keypoints2D_MPII[13] 
        l_elbow      = keypoints2D_MPII[14] 
        l_wrist      = keypoints2D_MPII[15]
        
        # spine = upper_thorax + pelvis
        # spine = spine / 2
        
        chest = upper_neck

        # Init H3.6M keypoints without nose
        keypoints2D_H36M = np.zeros((16,2))
        # Set array
        keypoints2D_H36M[0] = pelvis
        keypoints2D_H36M[1] = r_hip
        keypoints2D_H36M[2] = r_knee
        keypoints2D_H36M[3] = r_ankle
        keypoints2D_H36M[4] = l_hip
        keypoints2D_H36M[5] = l_knee
        keypoints2D_H36M[6] = l_ankle
        keypoints2D_H36M[7] = thorax
        keypoints2D_H36M[8] = chest
        # neck/nose ignored  
        keypoints2D_H36M[9] = head_top
        keypoints2D_H36M[10] = l_shoulder
        keypoints2D_H36M[11] = l_elbow
        keypoints2D_H36M[12] = l_wrist
        keypoints2D_H36M[13] = r_shoulder
        keypoints2D_H36M[14] = r_elbow
        keypoints2D_H36M[15] = r_wrist
        
        return keypoints2D_H36M
        
    def OpenPoseCOCOtoCOCO(self, keypoints2D_OpenPoseCOCO, output_format='MPII'):
        
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
        upper_neck = keypoints2D_OpenPoseCOCO[1]
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
        # Do not use
        #background = keypoints2D_OpenPoseCOCO[18]
        
        # Init array
        keypoints2D_COCO = np.zeros((17,2))
        # Set array
        keypoints2D_COCO[0]  = nose
        keypoints2D_COCO[1] = l_eye
        keypoints2D_COCO[2] = r_eye
        keypoints2D_COCO[3] = l_ear
        keypoints2D_COCO[4] = r_ear
        keypoints2D_COCO[5] = l_shoulder 
        keypoints2D_COCO[6] = r_shoulder 
        keypoints2D_COCO[7] = l_elbow 
        keypoints2D_COCO[8] = r_elbow 
        keypoints2D_COCO[9] = l_wrist 
        keypoints2D_COCO[10]= r_wrist 
        keypoints2D_COCO[11] = l_hip 
        keypoints2D_COCO[12] = r_hip 
        keypoints2D_COCO[13] = l_knee 
        keypoints2D_COCO[14] = r_knee 
        keypoints2D_COCO[15] = l_ankle 
        keypoints2D_COCO[16] = r_ankle 
        
        return keypoints2D_COCO
        
    def OpenPoseMPIItoMPII(self, keypoints2D_OpenPoseMPII):
        
        head_top = keypoints2D_OpenPoseMPII[0]
        upper_neck = keypoints2D_OpenPoseMPII[1]
        r_shoulder = keypoints2D_OpenPoseMPII[2]
        r_elbow = keypoints2D_OpenPoseMPII[3]
        r_wrist = keypoints2D_OpenPoseMPII[4]
        l_shoulder = keypoints2D_OpenPoseMPII[5]
        l_elbow = keypoints2D_OpenPoseMPII[6]
        l_wrist = keypoints2D_OpenPoseMPII[7]
        r_hip = keypoints2D_OpenPoseMPII[8]
        r_knee = keypoints2D_OpenPoseMPII[9]
        r_ankle = keypoints2D_OpenPoseMPII[10]
        l_hip = keypoints2D_OpenPoseMPII[11]
        l_knee = keypoints2D_OpenPoseMPII[12]
        l_ankle = keypoints2D_OpenPoseMPII[13]
        mid_thorax = keypoints2D_OpenPoseMPII[14]
        # Do not use
        background = keypoints2D_OpenPoseMPII[15]
        
        # Calculate pelvis
        pelvis = l_hip + r_hip
        pelvis = pelvis/2
        
        # Init MPII kpts
        keypoints2D_MPII = np.zeros((16,2))
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
        # pelvis = keypoints2D_OpenPoseMPII[8]/2 + keypoints2D_OpenPoseMPII[11]/2
        # self.keypoints2D_Baseline = np.array([keypoints2D_OpenPoseMPII[10], keypoints2D_OpenPoseMPII[9], keypoints2D_OpenPoseMPII[8],
        #                                         keypoints2D_OpenPoseMPII[11], keypoints2D_OpenPoseMPII[12], keypoints2D_OpenPoseMPII[13],
        #                                         pelvis, keypoints2D_OpenPoseMPII[15], keypoints2D_OpenPoseMPII[1],
        #                                         keypoints2D_OpenPoseMPII[0], keypoints2D_OpenPoseMPII[4], keypoints2D_OpenPoseMPII[3],
        #                                         keypoints2D_OpenPoseMPII[2], keypoints2D_OpenPoseMPII[5], keypoints2D_OpenPoseMPII[6],
        #                                         keypoints2D_OpenPoseMPII[15]])
    
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
        
        # Calculate upper_thorax
        upper_thorax = r_shoulder + l_shoulder
        upper_thorax = upper_thorax/2
        
        # Calculate head_top
        head_center = nose + r_ear + l_ear + r_eye + l_eye
        head_center = head_center/5
        vector_head = head_center - upper_thorax
        head_top = upper_thorax + 1.5*vector_head
        
        # Calculate upper_neck
        upper_neck_vec = head_top - upper_thorax
        upper_neck = upper_thorax + 0.3 * upper_neck_vec
        
        # Init MPII kpts
        keypoints2D_MPII = np.zeros((16,2))
        
        # Set array
        keypoints2D_MPII[0] = r_ankle
        keypoints2D_MPII[1] = r_knee
        keypoints2D_MPII[2] = r_hip
        keypoints2D_MPII[3] = l_hip
        keypoints2D_MPII[4] = l_knee
        keypoints2D_MPII[5] = l_ankle
        keypoints2D_MPII[6] = pelvis
        keypoints2D_MPII[7] = upper_thorax
        keypoints2D_MPII[8] = upper_neck
        keypoints2D_MPII[9] = head_top
        keypoints2D_MPII[10] = r_wrist
        keypoints2D_MPII[11] = r_elbow
        keypoints2D_MPII[12] = r_shoulder
        keypoints2D_MPII[13] = l_shoulder
        keypoints2D_MPII[14] = l_elbow
        keypoints2D_MPII[15] = l_wrist
        
        return keypoints2D_MPII
    
    def transformPersonsFromTo(self, persons2D_from, mode_from, mode_to):
        persons2D_to = []
        if mode_from == 'COCO' and mode_to == 'HM36M':
            for person2D in persons2D_from:
                person2D = self.COCOtoMPII(person2D)
                person2D = self.MPIItoHM36M(person2D)
                persons2D_to.append(person2D)
        elif mode_from == 'OpenPoseCOCO' and mode_to == 'HM36M':
            for person2D in persons2D_from:
                person2D = self.OpenPoseCOCOtoCOCO(person2D)
                person2D = self.COCOtoMPII(person2D)
                person2D = self.MPIItoHM36M(person2D)
                persons2D_to.append(person2D)
        else:
            print("Mode doesnt match")
        return np.array(persons2D_to)


    def preNormalizeKeypoints(self, image, keypoints2D):
        
        W = image.shape[1]
        H = image.shape[0]
        
        # COCO mode
        if(len(keypoints2D)==17):
            keypoints2D_pre_norm = np.zeros((17,2))
        
        # MPII mode
        if(len(keypoints2D)==16):
            keypoints2D_pre_norm = np.zeros((16,2))
        
        keypoints2D_pre_norm[:, 0] = keypoints2D[:, 0] / W
        keypoints2D_pre_norm[:, 1] = keypoints2D[:, 1] / H
        
        return keypoints2D_pre_norm
    
    def getUpperBody(self, keypoints):
        
        return keypoints[self.pos_upperbody]
    
    