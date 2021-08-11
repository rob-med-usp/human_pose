import cv2
import numpy as np
import os

class Deprojector:
    
    def loadCameraIntrinsics(self, camera_str):
        
        fn = camera_str + '_intrinsics.npy'
        path = os.path.join('src', 'cameras', fn)
        with open(path, 'rb') as f:
            mtx = np.load(f)
        
        return mtx
    
    def loadCameraDistortion(self, camera_str):
        
        fn = camera_str + '_distortion.npy'
        path = os.path.join('src', 'cameras', fn)
        with open(path, 'rb') as f:
            dist = np.load(f)
        
        return dist
    
    def deprojectPose(self, keypoints2D, keypoints3D_local):
        
        intrinsics = self.loadCameraIntrinsics('webcam_acer')
        distortion = self.loadCameraDistortion('webcam_acer')
        
        # Remove nose from keypoints3D
        keypoints3D_local = np.delete(keypoints3D_local, 9, 0)
        
        print(f'Kpts 3D shape: {keypoints3D_local.shape}')
        print(f'Kpts 2D shape: {keypoints2D.shape}')
        print(f'Mtx shape: {intrinsics.shape}')
        print(f'Distort shape: {distortion.shape}')
        
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(keypoints3D_local, keypoints2D, intrinsics, distortion)
        
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intrinsics, distortion)
        
        # Translate points
        keypoints3D_global = keypoints3D_local + tvecs.T
        
        return keypoints3D_global, rvecs, tvecs, imgpts