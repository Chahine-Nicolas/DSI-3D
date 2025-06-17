import numpy as np
import json

def save_GT(path_bin, sequence_path, eval_seq, revisit_json, positions_database, timestamps, skip_time=30):
    dictio = {}
    is_revisit_list = revisit_json[eval_seq]
    
    for i in range (len(path_bin)):

        lidar_path = path_bin[i]
        query_idx = int(path_bin[i].split('/')[-1][:-4])

        if is_revisit_list[query_idx] == 1.0:
            query_pose = positions_database[query_idx]
            query_time = timestamps[query_idx]
            d = []
            q = []
            for comp_idx in range(len(path_bin)):
                comp_time = timestamps[comp_idx]

                if (abs(query_time - comp_time) - skip_time) < 0:
                        continue
                if query_idx != comp_idx:
                    comp_pose = positions_database[comp_idx]
                    p_dist = np.linalg.norm(query_pose - comp_pose)
                    d.append(round(p_dist,3))
                    q.append( '%06d' % round(comp_idx,3))

            d, q = zip(*sorted(zip(d, q)))

            dictio['%06d' % query_idx] = [q[:10], d[:10]]
            
    with open(sequence_path + "matching.json", 'w', encoding ='utf8') as json_file: 
        json.dump(dictio, json_file, allow_nan=False) 
    print("saved at ", sequence_path + 'matching.json') 
    return


"""
##################################################################################
EXEMPLE
##################################################################################

import numpy as np
import glob
import os
import torch
import re
import json
import glob

import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
#eval_arg.add_argument('--eval_pipeline', type=str, default='PointNetVLAD')
eval_arg.add_argument('--kitti_eval_seq', type=int, default=2)
eval_arg.add_argument('--mulran_eval_seq', type=str,default='DCC/DCC_01')
#eval_arg.add_argument('--checkpoint_name', type=str,default='/kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth')
eval_arg.add_argument('--checkpoint_name', type=str,default='/logg_epoc_35_kitti')
#eval_arg.add_argument('--checkpoint_name', type=str,default='/logg_epoc_31_mulran')
eval_arg.add_argument('--eval_batch_size', type=int, default=1)
eval_arg.add_argument('--test_num_workers', type=int, default=3)
eval_arg.add_argument("--eval_random_rotation", type=str2bool,default=False, help="If random rotation. ")
eval_arg.add_argument("--eval_random_occlusion", type=str2bool,default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=3, type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")


# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset
data_arg.add_argument('--eval_dataset', type=str, default = 'KittiDataset')
#data_arg.add_argument('--eval_dataset', type=str, default = 'MulRanDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=False)
data_arg.add_argument("--eval_save_counts", type=str2bool, default=False)
data_arg.add_argument("--eval_plot_pr_curve", type=str2bool, default=True)
data_arg.add_argument('--num_points', type=int, default=80000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine #euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/gpfswork/rech/dki/ujo91el/datas/datasets/',help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    'val': [],
    'test': [8]
})

data_arg.add_argument('--mulran_dir', type=str,default='/gpfswork/rech/dki/ujo91el/datas/mulran/', help="Path to the MulRan dataset")

data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,default=False, help="Normalize intensity return.")

data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})


# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


def get_config_eval():
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)

#####################################################################################
# Load poses
#####################################################################################

def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    return Tcam @ cam2velo

def load_poses_from_txt(file_name):
    
    #Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
        
    return transforms, np.asarray(positions)

#####################################################################################
# Load timestamps
#####################################################################################

def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn


eval_seq = 6
eval_seq = '%02d' % eval_seq

kitti_infos = glob.glob("/gpfswork/rech/dki/ujo91el/datas/datasets/sequences/"+eval_seq +"/velodyne/*.bin")


sequence_path = '/gpfswork/rech/dki/ujo91el/datas/datasets/sequences/'+eval_seq+'/'
#poses
_, positions_database = load_poses_from_txt(
        sequence_path + 'poses.txt')

positions_database = positions_database

#time
timestamps = load_timestamps(sequence_path + 'times.txt')

#revisit
revisit_criteria = 3
skip_time = 30

revisit_json_dir = '/gpfswork/rech/dki/ujo91el/code/logg3dnet/config/kitti_tuples/'
revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(int(revisit_criteria), int(skip_time))
revisit_json = json.load(open(revisit_json_dir + revisit_json_file, "r"))


save_GT(kitti_infos, sequence_path, eval_seq, revisit_json, positions_database, timestamps, skip_time)
"""
