import numpy as np

class Fusion:

    def mergeResults(self, kpts_stereo, kpts_mlp, thresh = 500):

        assert(kpts_stereo.shape == kpts_mlp.shape)
        
        # Filter points that are far than 2 meters of the centroid
        kpts_stereo = self.filterStereoKpts(kpts_stereo, 2000)

        # Translate mlp kpts to stereo centroid
        kpts_mlp = self.translateMLP(kpts_mlp, kpts_stereo)

        # Compare MLP and Stereo results 
        # if kpts mlp - stereo < comp_thresh: kpts_stereo is used
        # else kpts mlp is used
        # also if kpts_stereo is [-1, -1, -1], mlp is used
        num_kpts = len(kpts_stereo)
        kpts_merged = np.ones((num_kpts, 3))*(-1)

        dists = self.distance(kpts_stereo, kpts_mlp)
        for idx, (_, dist) in enumerate(zip(kpts_merged, dists)):
            if dist < thresh:
                kpts_merged[idx] = kpts_stereo[idx]
            else:
                kpts_merged[idx] = kpts_mlp[idx]

        return kpts_merged
    
    def distance(self, pt1, pt2):
        # ex = abs(pts1[:,0] - pts2[:,0])
        # ey = abs(pts1[:,1] - pts2[:,1])
        # ez = abs(pts1[:,2] - pts2[:,2])
        # dist = np.sqrt(ex**2 + ey**2 + ez**2)
        dist = np.sqrt(np.sum(np.power(pt1-pt2,2), axis=1))
        return dist

    def filterStereoKpts(self, kpts_stereo, global_thresh):

        ### Global filter
        centroid = np.mean(kpts_stereo, axis=0, where=(kpts_stereo!=[-1,-1,-1]))
        
        # kpts global filtered
        kpts_stereo_cpy = kpts_stereo.copy()

        dists = self.distance(kpts_stereo, centroid)
        for idx, kpt in enumerate(kpts_stereo):
            dist = dists[idx]
            if dist < global_thresh:
                kpts_stereo_cpy[idx] = kpts_stereo[idx]
            else:
                kpts_stereo_cpy[idx] = [-1, -1, -1]

        ### Local filter
        #TODO  

        return kpts_stereo_cpy
    
    def chooseMLPStereo(self, kpts_mlp, kpts_stereo, comp_thresh):
        dists = self.distance(kpts_mlp, kpts_stereo)
        kpts_merged = np.where((dists < comp_thresh).T and ((kpts_stereo != [-1,-1,-1])).any(), kpts_stereo, kpts_mlp)

        return kpts_merged

    def translateMLP(self, kpts_mlp, kpts_stereo):
        
        # Centroids
        stereo_centroid = np.mean(kpts_stereo, axis=0, where=(kpts_stereo!=[-1,-1,-1]))
        mlp_centroid = np.mean(kpts_mlp, axis=0)

        # Calculate translation
        t = stereo_centroid - mlp_centroid

        # Translate all points using numpy broadcast
        kpts_mlp_trans = kpts_mlp + t

        return kpts_mlp_trans

'''
num_kpts = len(kpts_stereo)

        kpts_merged = np.ones((num_kpts, 3))

        for i in range(len(kpts_stereo)):
            ex = abs(kpts_stereo[i][0] - kpts_mlp[i][0])
            ey = abs(kpts_stereo[i][1] - kpts_mlp[i][1])
            ez = abs(kpts_stereo[i][2] - kpts_mlp[i][2])

            dist = np.sqrt(ex**2 + ey**2 + ez**2)
'''