import numpy as np
import os

path = os.path.join('results_tmp','mean_error_12-5_10:39.npy')

with open(path, 'rb') as f:
    mean_error = np.load(f)

print(f"Analysing {mean_error.shape[0]} annotations:")

print(f"Mean error: {np.mean(mean_error)} mm")


path = os.path.join('results_tmp','error_per_joint_12-5_10:39.npy')

with open(path, 'rb') as f:
    error_per_joint = np.load(f)

print(f"Mean Error Per Joint:")
print(np.mean(error_per_joint, axis=0))
