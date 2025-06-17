import os
import torch
import numpy as np
from scipy.spatial.distance import cdist

def load_tensor(sequence_path, nn,lab) :
    
    xx = []
    for lli in nn : 
        #seq = os.path.splitext(lli)[0][:2]
        file = os.path.splitext(lli)[0]#[2:]
        fname = sequence_path + lab + "/" + (file +'.pt')
        if os.path.isfile(fname) : 
            xx1 = torch.load(fname)
            xx.append(xx1)
        else :
            print("TENSOR NOT FOUND")
            print(fname)
            return None
    return torch.stack(xx)

import os
import matplotlib.pyplot as plt 
from module_loader_kitti_pose import * 

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
timestamps = load_timestamps(sequence_path + 'times.txt')

def get_desc(sequence_path, query,lab):
    d = load_tensor(sequence_path, query,lab).cpu().detach().numpy()
    d = np.reshape(d , (1, -1))
    return d

def getdist_tfilter(sequence_path, query, lab, metrics = 'cosine' ):
    time = []
    dist = []
    dist_geom = []
    query_nn = np.array([ '%06d' % query + '.bin'], dtype='<U10')
    query_nn = load_tensor(sequence_path, query_nn, lab).cpu().detach().numpy()
    query_nn   =  np.reshape(query_nn  , (1, -1))

    start_time = timestamps[query]
    skip_time = 30 
    
    for i in range(4541):
        time.append(i)
        query_time = timestamps[i]
        
        place_candidate = pose[query]
        place_candidate2 = pose[i]
        p_dist = np.linalg.norm(place_candidate - place_candidate2)
        dist_geom.append(p_dist)
        
        if start_time - skip_time <= query_time <= start_time + skip_time:
            #print("WARNING CONTINUE COMMENT : REMOVE FOR EVAL")
            dist.append(np.array([1]))
            continue
            
        
        
        query_nn2 = np.array([ '%06d' % i + '.bin'], dtype='<U10')
        comp_nn = get_desc(sequence_path, query_nn2, lab)
        
        dist.append(cdist(query_nn, comp_nn, metric=metrics).reshape(-1))
    
    return time, dist, dist_geom

pt_logg = 0
pt_sop = 0

for i in range(290, 4541):
    query = i
    query_pose = pose[i]

    print('is revisited ', is_revisit_list[i])
    
    lab = "logg_desc"
    time, dist_impl, dist_geom = getdist_tfilter(sequence_path, query, lab , metrics = 'cosine' )
    
    
    place_candidate = pose[np.argmin(dist_impl)]
    p_dist = np.linalg.norm(query_pose - place_candidate)
    print(i, lab, min(dist_impl), np.argmin(dist_impl), p_dist)
    if is_revisit_list[i] == 1.0 and p_dist <= 3:
        pt_logg += 1
    
    lab = "sop"
    time,  dist_impl, dist_geom = getdist_tfilter(sequence_path, query, lab, metrics = 'cosine' )
    
    place_candidate = pose[np.argmin(dist_impl)]
    p_dist = np.linalg.norm(query_pose - place_candidate)
    print(i, lab, min(dist_impl), np.argmin(dist_impl), p_dist)

    if is_revisit_list[i] == 1.0 and p_dist <= 3:
        pt_sop += 1


    print('pt ', pt_logg, pt_sop)
