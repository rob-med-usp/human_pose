# %%

import numpy as np
import json
import cv2
import json
import os
import random

import matplotlib.pyplot as plt

from src.HydraPose import HydraPose, SEFFPOSE
from src.SkeletonsBridge import SkeletonsBridge
from src.Visualizer import Visualizer
bridge = SkeletonsBridge()


import sys
sys.path.insert(0, '/home/guisoares/soares_repo/MVOR/lib/')

from visualize_groundtruth import create_index, viz2d, plt_imshow, bgr2rgb, plt_3dplot, coco_to_camma_kps, progress_bar


# %% [markdown]
# Some functions to make the evaluation process

# %%

def hydrapose(img_path):
    img = cv2.imread(img_path)
    persons = hy.estimate3DPose()
    return persons

def getCamMtxFromDataset(data_dict, id):

    # Get focal values
    fx = data_dict['cameras_info']['camParams']['intrinsics'][id]['focallength'][0]
    fy = data_dict['cameras_info']['camParams']['intrinsics'][id]['focallength'][1]

    # Get center values
    cx = data_dict['cameras_info']['camParams']['intrinsics'][id]['principalpoint'][0]
    cy = data_dict['cameras_info']['camParams']['intrinsics'][id]['principalpoint'][1]

    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
    return mtx

def getCamDistFromDataset(data_dict, id):

    # Get distortion array
    dist = np.array(data_dict['cameras_info']['camParams']['intrinsics'][id]['distortion'])
    return dist

def getPersonMediumDist(person1, person2):
    dists = []
    for kpt1, kpt2 in zip(person1, person2):
        dist = np.sqrt(np.sum(np.power(kpt1 - kpt2, 2)))
        dists.append(dist)
    
    dists = np.array(dists)
    mean = np.mean(dists)
    return mean, dists

def getMinimalDist(persons_est, persons_annon):

    # Store
    min_index_list = []
    min_error_per_joint_list = []
    min_mean_error_per_joint_list = []
    # Get the annotation index that results the minimal mean error
    for person_est in persons_est:
        mean_error_list = []
        error_per_joint_list = []
        for person_annon in persons_annon:
            mean_error, error_per_joint = getMinimalDist(person_est, person_annon)
            mean_error_list.append(mean_error)
            error_per_joint_list.append(error_per_joint)
        # Get the index of the minimum error and insert on a list that contains it 
        min_index = np.argmin(mean_error_list)
        min_index_list.append(min_index)
        min_error_per_joint_list.append(error_per_joint_list[min_index])
        min_mean_error_per_joint_list.append(mean_error_list[min_index])
    
    # Certifie that index list has diferent elements, meaning that no estimation has the same annotation
    _, count = np.unique(min_index_list, return_counts = True)
    for c in count:
        if c != 1:
            raise AssertionError("The mapped index list has duplicated elements")

    # Remap annotation to
    persons_annon_remapped = persons_annon[min_index_list]

    return persons_est, persons_annon_remapped, min_mean_error_per_joint_list, min_error_per_joint_list

# %%
GT_ANNO_PATH = os.path.join(os.path.expanduser('~'), "soares_repo", "MVOR","annotations/camma_mvor_2018.json")
GT_IMGS_PATH = '/media/guisoares/guisoares-ext-hdd/Datasets/camma_mvor_dataset/'

# load the ground truth annotations
camma_mvor_gt = json.load(open(GT_ANNO_PATH))
anno_2d, anno_3d, mv_paths, imid_to_path = create_index(camma_mvor_gt)


# %%
# Read a random multi-view image
# imid_3d = random.choice(list(mv_paths.keys()))
imid_3d = "10010000013_10020000013_10030000013"
imids_2d = [int(m) for m in imid_3d.split("_")]
imgs = [cv2.imread(os.path.join(GT_IMGS_PATH, imid_to_path[_p])) for _p in imids_2d]

anns2d = [anno_2d[str(ann)] for ann in imids_2d]
# anno_3d[imgs_ids][person]['keypoint3D']
anns3d = anno_3d[imid_3d]

#anns2d[img][person]['keypoints']
persons2D_ann = []
for i in range(len(anns2d[0])):
    persons2D_ann.append(anns2d[0][i]['keypoints'])
persons2D_ann = np.array(persons2D_ann)
persons2D_ann = persons2D_ann.reshape(-1,10,3)
persons2D_ann = persons2D_ann[:,:,:2]

persons3D_ann = []
for i in range(len(anns3d)):
    persons3D_ann.append(anns3d[i]['keypoints3D'])
persons3D_ann = np.array(persons3D_ann)
persons3D_ann = persons3D_ann.reshape(-1,12,4)
persons3D_ann = persons3D_ann[:,:,:3]

# %%
anns3d[0]['keypoints3D']

# %%
print(len(anns2d[1][2]['keypoints']))
print(len(anns3d[0]['keypoints3D']))

# %% [markdown]
# There is some garbage on annotations. In this case all jeypoints are 0 and person_id = -1

# %%
anns2d[0][5]

# %%
def runHydra():
    hy = HydraPose(pose3D = SEFFPOSE)
    # hy.setIntrinsics(getCamMtxFromDataset(camma_mvor_gt,0),getCamDistFromDataset(camma_mvor_gt,0))
    hy.setIntrinsics(getCamMtxFromDataset(camma_mvor_gt,0), np.array([0.,0.,0.,0.,0.]))
    persons = hy.estimate3DPose(imgs[0])
    persons = persons[:4]
    print(persons.shape)
    persons_aux = np.zeros((persons.shape[0],10,3))
    for idx, person in enumerate(persons):
        persons_aux[idx] = bridge.HM36MtoMVOR(person)
    hy.initWindow()
    hy.viz.ax3D.set_xlim(-1000, 1000)
    hy.viz.ax3D.set_ylim(-1000, 1000)
    hy.viz.ax3D.set_zlim(-200, 2200)
    hy.persons3DHybrid = np.array(persons_aux[1:2])
    hy.plotPersons(imgs[0].copy(), block=False, mode ='MVOR')
    return persons_aux
persons = runHydra()

# %%
persons[1,:10]

# %% [markdown]
# Simulando o plot das anotações

# %%
persons3D_ann

# %%

print(persons3D_ann.shape)
def simulateAnnonPlot():
    hy = HydraPose(pose3D = SEFFPOSE)
    hy.persons2D = persons2D_ann[:4]
    hy.persons3DHybrid = persons3D_ann[:4]
    hy.initWindow()
    hy.viz.ax3D.set_xlim(-1000, 1000)
    hy.viz.ax3D.set_ylim(-1000, 1000)
    hy.viz.ax3D.set_zlim(-200, 2200)
    hy.plotPersons(imgs[0].copy(), block=False, mode='MVOR')
simulateAnnonPlot()

id_ann = 0
id = 1

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(persons[:,:10,0],persons[:,:10,1],persons[:,:10,2])
# ax.scatter(persons3D_ann[:,:10,0],persons3D_ann[:,:10,1],persons3D_ann[:,:10,2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_xlim(-1000, 1000)
# ax.set_ylim(-1000, 1000)
# ax.set_zlim(-200, 2200)
# ax.azim = -90
# ax.dist = 10
# ax.elev = -60
# plt.show()

fig = plt.figure()
ax3D = fig.add_subplot(1, 1, 1, projection = '3d')
# Set right visualization position 
ax3D.azim = -90
ax3D.dist = 10
ax3D.elev = -60
# space around the persons
RADIUS = 1500 
ax3D.set_xlim([-RADIUS, RADIUS])
ax3D.set_ylim([-RADIUS ,RADIUS])
ax3D.set_zlim([0, 2*RADIUS])

ax3D.set_xlabel('x')
ax3D.set_ylabel('y')
ax3D.set_zlabel('z')
viz = Visualizer()
for person in persons[:4,:10]:
    ax3D = viz.drawBones3D(ax3D, person, bridge.pairs_MVOR, color=viz.camma_colors_skeleton)
for person in persons3D_ann[:4,:10]:
    ax3D = viz.drawBones3D(ax3D, person, bridge.pairs_MVOR)
plt.show()

print(persons3D_ann[0,:10] - persons[0,:10])
input()

