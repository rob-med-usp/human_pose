import numpy as np

class Fusion:

    def mergeResults(self, kptsRGBD, kptsMLP, tresh = 200):

        assert(kptsRGBD.shape == kptsMLP.shape)
        
        num_kpts = len(kptsRGBD)

        kpts_merged = np.ones((num_kpts, 3))

        for i in range(len(kptsRGBD)):
            ex = abs(kptsRGBD[i][0] - kptsMLP[i][0])
            ey = abs(kptsRGBD[i][1] - kptsMLP[i][1])
            ez = abs(kptsRGBD[i][2] - kptsMLP[i][2])

            dist = np.sqrt(ex**2 + ey**2 + ez**2)

            if dist >= tresh | kptsRGBD[i] == [-1,-1,-1]:
                kpts_merged[i] = kptsMLP[i]
            else:
                kpts_merged[i] = kptsRGBD[i]
        
        return kpts_merged
