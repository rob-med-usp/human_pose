import numpy as np
import json
import cv2
import json
import time
import os
import random

import matplotlib.pyplot as plt

from src.HydraPose import HydraPose, SEFFPOSE
from src.SkeletonsBridge import SkeletonsBridge
from src.Fusion import Fusion
from src.Visualizer import Visualizer
bridge = SkeletonsBridge()


import sys
sys.path.insert(0, '/home/guisoares/soares_repo/MVOR/lib/')

from visualize_groundtruth import create_index, viz2d, plt_imshow, bgr2rgb, plt_3dplot, coco_to_camma_kps, progress_bar


def getCamMtxFromDataset(data_dict, idx):

    # Get focal values
    fx = data_dict['cameras_info']['camParams']['intrinsics'][idx]['focallength'][0]
    fy = data_dict['cameras_info']['camParams']['intrinsics'][idx]['focallength'][1]

    # Get center values
    cx = data_dict['cameras_info']['camParams']['intrinsics'][idx]['principalpoint'][0]
    cy = data_dict['cameras_info']['camParams']['intrinsics'][idx]['principalpoint'][1]

    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])
    return mtx

def getCamDistFromDataset(data_dict, id):

    # Get distortion array
    dist = np.array(data_dict['cameras_info']['camParams']['intrinsics'][id]['distortion'])
    return dist

def getPersonMediumDist(person1, person2):

    error_per_joint = []
    for kpt1, kpt2 in zip(person1, person2):
        error = np.sqrt(np.sum(np.power(kpt1 - kpt2, 2)))
        error_per_joint.append(error)
    
    error_per_joint = np.array(error_per_joint)
    mean = np.mean(error_per_joint)
    return mean, error_per_joint

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
            mean_error, error_per_joint = getPersonMediumDist(person_est, person_annon)
            mean_error_list.append(mean_error)
            error_per_joint_list.append(error_per_joint)
        # Get the index of the minimum error and insert on a list that contains it 
        min_index = np.argmin(mean_error_list)
        min_index_list.append(min_index)
        min_error_per_joint_list.append(error_per_joint_list[min_index])
        min_mean_error_per_joint_list.append(mean_error_list[min_index])
    
    # Certifie that index list has diferent elements, meaning that no estimation has the same annotation
    
    # _, count = np.unique(min_index_list, return_counts = True)
    # for c in count:
    #     if c != 1:
    #         raise AssertionError("The mapped index list has duplicated elements")

    # Remap annotation to
    persons_annon_remapped = persons_annon[min_index_list]
    min_mean_error_per_joint_arr = np.array(min_mean_error_per_joint_list)
    min_error_per_joint_arr = np.array(min_error_per_joint_list)
    min_error_per_joint_arr = np.reshape(min_error_per_joint_arr, (-1,10))
    return persons_est, persons_annon_remapped, min_mean_error_per_joint_arr, min_error_per_joint_arr

def getAnnotations(GT_IMGS_PATH, imid_3d, imid_to_path, anno_2d, anno_3d, cam):

    imids_2d = [int(m) for m in imid_3d.split("_")]
    anns2d = [anno_2d[str(ann)] for ann in imids_2d]
    anns3d = anno_3d[imid_3d]

    persons2D_ann = []
    # anns2d[camera]
    for i in range(len(anns2d[cam])):
        persons2D_ann.append(anns2d[0][i]['keypoints'])
    persons2D_ann = np.array(persons2D_ann)
    persons2D_ann = persons2D_ann.reshape(-1,10,3)
    persons2D_ann = persons2D_ann[:,:,:2]

    persons3D_ann = []
    for i in range(len(anns3d)):
        persons3D_ann.append(anns3d[i]['keypoints3D'])
    persons3D_ann = np.array(persons3D_ann)
    # WTF 12 JOINTS??????????
    persons3D_ann = persons3D_ann.reshape(-1,12,4)
    persons3D_ann = persons3D_ann[:,:10,:3]

    # Read a random multi-view image
    # imid_3d = random.choice(list(mv_paths.keys()))
    # imid_3d = "10010000013_10020000013_10030000013"
    
    rgb_paths = [imid_to_path[img_id] for img_id in imids_2d]
    depth_paths = [rgb_path.replace('color', 'depth') for rgb_path in rgb_paths]

    rgb_imgs = [cv2.imread(os.path.join(GT_IMGS_PATH, rgb_path)) for rgb_path in rgb_paths]
    depth_imgs = [cv2.imread(os.path.join(GT_IMGS_PATH, depth_path), cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]

    return rgb_imgs, depth_imgs, persons2D_ann, persons3D_ann

def runHydraSeffPose(imgs, cam, camma_mvor_gt):
    hy = HydraPose(pose3D = SEFFPOSE)
    # hy.setIntrinsics(getCamMtxFromDataset(camma_mvor_gt,0),getCamDistFromDataset(camma_mvor_gt,0))
    hy.setIntrinsics(getCamMtxFromDataset(camma_mvor_gt,0), np.array([0.,0.,0.,0.,0.]))
    persons = hy.estimate3DPose(imgs[0])
    if len(persons)==0:
        return np.array([])
    persons_aux = np.zeros((persons.shape[0],10,3))
    for idx, person in enumerate(persons):
        persons_aux[idx] = bridge.HM36MtoMVOR(person)
    
    return persons_aux

def getPersonFromDepth(cam_mtx, depth_img, person2D):
    # Get focal distances
    fx = cam_mtx[0][0]
    fy = cam_mtx[1][1]
    
    # Get principal points
    cx = cam_mtx[0][2]
    cy = cam_mtx[1][2]

    person_stereo = np.zeros((person2D.shape[0],3))
    for idx, [u,v] in enumerate(person2D):
        u, v = int(u), int(v)
        depth = depth_img[v][u]
        if depth <= 0:
            person_stereo[idx] = [-1, -1, -1]
        else:
            x = depth*(u - cx)/fx
            y = depth*(v - cy)/fy
            z = depth
            person_stereo[idx] = [x, y, z]
    
    person_stereo = bridge.HM36MtoMVOR(person_stereo)
        
    return person_stereo

def runHydra(rgb_imgs, depth_imgs, cam, camma_mvor_gt):
    hy = HydraPose(pose3D = SEFFPOSE)
    fus = Fusion()
    
    cam_mtx = getCamMtxFromDataset(camma_mvor_gt,0)
    # SeffPose
    # hy.setIntrinsics(getCamMtxFromDataset(camma_mvor_gt,0),getCamDistFromDataset(camma_mvor_gt,0))
    hy.setIntrinsics(cam_mtx, np.array([0.,0.,0.,0.,0.]))
    persons = hy.estimate3DPose(rgb_imgs[0])
    
    # Checking number of persons
    number_of_persons = len(persons)
    if number_of_persons == 0:
        return np.array([])
    
    # From HM36M to MVOR
    persons_seff = np.zeros((persons.shape[0], 10, 3))
    for idx, person in enumerate(persons):
        persons_seff[idx] = bridge.HM36MtoMVOR(person)
    
    # StereoPose
    persons_stereo = np.zeros((persons.shape[0], 10, 3))
    for idx, person2D in enumerate(hy.persons2D):
        persons_stereo[idx] = getPersonFromDepth(cam_mtx, depth_imgs[0], person2D)
    
    # Fusion
    persons_hy = np.zeros((persons.shape[0], 10, 3))
    for idx in range(len(persons_stereo)):
        persons_hy[idx] = fus.mergeResults(persons_stereo[idx], persons_seff[idx])
    
    return persons_hy

def simulateAnnonPlot(imgs, persons2D_ann, persons3D_ann):
    hy = HydraPose(pose3D = SEFFPOSE)
    hy.persons2D = persons2D_ann[:4]
    hy.persons3DHybrid = persons3D_ann[:4]
    hy.initWindow()
    hy.viz.ax3D.set_xlim(-1000, 1000)
    hy.viz.ax3D.set_ylim(-1000, 1000)
    hy.viz.ax3D.set_zlim(-200, 2200)
    hy.plotPersons(imgs[0].copy(), block=False, mode='MVOR')

def comparePlot3D(img, persons1, persons2):
    fig = plt.figure()
    # Plot image
    ax = fig.add_subplot(1,2,1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax3D = fig.add_subplot(1, 2, 2, projection = '3d')
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
    for person in persons1:
        ax3D = viz.drawBones3D(ax3D, person, bridge.pairs_MVOR, color=viz.camma_colors_skeleton)
    for person in persons2:
        ax3D = viz.drawBones3D(ax3D, person, bridge.pairs_MVOR)
    plt.show()

    return fig

def saveCurrState(mean_error, error_per_joint):

    t = time.localtime()
    day = t.tm_mday
    month = t.tm_mon
    hour = t.tm_hour
    mins = t.tm_min
    print(f'Finished evaluating at time {day}/{month}-{hour}:{mins}')

    path = os.path.join('results',f'mean_error_{day}-{month}_{hour}:{mins}.npy')
    with open(path, 'wb') as f:
        np.save(f, mean_error)

    path = os.path.join('results',f'error_per_joint_{day}-{month}_{hour}:{mins}.npy')
    with open(path, 'wb') as f:
        np.save(f, error_per_joint)

def main():

    GT_ANNO_PATH = os.path.join(os.path.expanduser('~'), "soares_repo", "MVOR", "annotations/camma_mvor_2018.json")
    GT_IMGS_PATH = '/media/guisoares/guisoares-ext-hdd/Datasets/camma_mvor_dataset/'

    # load the ground truth annotations
    camma_mvor_gt = json.load(open(GT_ANNO_PATH))

    anno_2d, anno_3d, mv_paths, imid_to_path = create_index(camma_mvor_gt)

    imids_3d = list(mv_paths.keys())
    len_ids = len(imids_3d)

    mean_error = np.empty((0))
    error_per_joint = np.empty((0,10))

    elapsed_time = 0
    # id with 2 persons but only one annon: '10010000005_10020000005_10030000005'
    try:
        for i, imid_3d in enumerate(imids_3d):
            os.system('clear')
            print(f"Status: i = {i} total = {len_ids} progress = {round((i*100)/len_ids,3)}%")
            print(f"Last elapsed time: {round(elapsed_time,1)} s. Estimated time: {round((len_ids-i)*elapsed_time/3600,2)} h.")
            print(f"Evaluating id={imid_3d}...")

            start = time.time()
            rgb_imgs, depth_imgs, persons2D_ann, persons3D_ann = getAnnotations(GT_IMGS_PATH, imid_3d, imid_to_path, anno_2d, anno_3d, 0)

            if persons3D_ann.shape[0] == 0:
                print("No annotation, going to next id...")
                continue

            persons = runHydra(rgb_imgs, depth_imgs, 0, camma_mvor_gt)
            
            if persons.shape[0] == 0:
                print("No estimatives, going to next id...")
            
            # Simulating plots from annontations
            print(persons3D_ann.shape)
            # simulateAnnonPlot(imgs, persons2D_ann, persons3D_ann)

            # Comparing the results
            # fig = comparePlot3D(imgs[0], persons, persons3D_ann)

            persons, persons3D_ann, mean_error_per_joint_arr, error_per_joint_arr = getMinimalDist(persons, persons3D_ann)
            
            mean_error = np.hstack((mean_error, mean_error_per_joint_arr))
            error_per_joint = np.vstack((error_per_joint, error_per_joint_arr))
            print(error_per_joint_arr)
            print(mean_error_per_joint_arr)

            elapsed_time = time.time() - start

            # input()
            # plt.close(fig)
            # input()
        
        saveCurrState(mean_error, error_per_joint)

    finally:
        saveCurrState(mean_error, error_per_joint)
        print(f"Saved array. Quiting.")

if __name__ == '__main__':
    main()
