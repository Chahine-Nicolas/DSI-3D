#############################################
#import 
#############################################
import os
import sys
import torch
import logging
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pickle
import numpy as np
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from config.test_config import get_config_test
from models.pipeline_factory import get_pipeline
from models.pipelines.pipeline_utils import *

#############################################
#paramètres testés
#############################################

#perturbations
tremblement = False
rotation = False
occlusion = True
crop = False
miroir = False

#query
query_idx = 5493


def test_rotation(xyzr, r_angle=360):
    # Rotate about z-axis by fixed angle 'r_angle'.
    r_angle = (np.pi/180) * r_angle
    print("r_angle", r_angle)
    cos_angle = np.cos(r_angle)
    sin_angle = np.sin(r_angle)
    rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                           [sin_angle, cos_angle, 0],
                           [0,             0,      1]])
    scan = xyzr[:, :3]
    #print("pcd_data.py: xyzr shape",np.shape(xyzr),xyzr[0])
    int = xyzr[:, 3].reshape((-1, 1))
    augmented_scan = np.dot(scan, rot_matrix)
    augmented_scan = np.hstack((augmented_scan, int))
    return augmented_scan.astype(np.float32)


def test_bruit(xyzr, n_sigma = 0.01):
    scan = xyzr[:, :3]
    #print("pcd_data.py: xyzr shape",np.shape(xyzr),xyzr[0])
    int = xyzr[:, 3].reshape((-1, 1))
    r_angle = (np.pi/180) * 360
    print("n_sigma", n_sigma)
    cos_angle = np.cos(r_angle)
    sin_angle = np.sin(r_angle)
    rot_matrix = np.array([[cos_angle, -sin_angle, 0],
                           [sin_angle, cos_angle, 0],
                           [0,             0,      1]])
    augmented_scan = np.dot(scan, rot_matrix)

    # Add gaussian noise
    noise = np.clip(n_sigma * np.random.randn(*
                    augmented_scan.shape), -3*n_sigma, 3*n_sigma)
    augmented_scan = augmented_scan + noise

    augmented_scan = np.hstack((augmented_scan, int))
    return augmented_scan.astype(np.float32)


def occlude_scan(xyzr, angle=30):
    # Remove points within a sector of fixed angle (degrees) and random heading direction.
    thetas = (180/np.pi) * np.arctan2(xyzr[:, 1], xyzr[:, 0])
    heading = (180-angle/2)*np.random.uniform(-1, 1)
    occ_scan = np.vstack(
        (xyzr[thetas < (heading - angle/2)], xyzr[thetas > (heading + angle/2)]))
    return occ_scan.astype(np.float32)



#############################################
#dataloading
#############################################

# Get config
cfg = get_config_test()

#Adapt name
if 'Kitti' in cfg.eval_dataset:
    eval_seq = cfg.kitti_eval_seq
    eval_seq = '%02d' % eval_seq

elif 'MulRan' in cfg.eval_dataset:
    eval_seq = cfg.mulran_eval_seq
    
    #check name for mulran
    name_file = open("/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/classification/"+str(cfg.mulran_eval_seq)+"_Names_MulRanDataset.pkl", "rb")
    name_dict = pickle.load(name_file)

seq = str(eval_seq)
dat = str(cfg.eval_dataset)


#check descriptor.pkl
descriptor_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/"+seq+"/logg3d_descriptor.pickle"
print("descriptor_path", descriptor_path)
input_file = open(descriptor_path, "rb")
seen_descriptors = pickle.load(input_file)
db_seen_descriptors = np.copy(seen_descriptors)


#check pairing_TP.pkl
TP_file = open("/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/classification/"+seq+'_seq_TP_dataset_'+dat+'.pkl', "rb")
print("input_file", TP_file)
dict_TP = pickle.load(TP_file)
TP_idx = dict_TP[query_idx] 
print("TP_idx", TP_idx)

#nuage
og_name = name_dict[query_idx]
if 'Kitti' in cfg.eval_dataset:
    fname = cfg.kitti_dir + 'sequences/'+seq+'/velodyne/'+'%06d' % query_idx + '.bin'

elif 'MulRan' in cfg.eval_dataset:
    fname = '/gpfswork/rech/dki/ujo91el/datas/mulran/'+seq+'/Ouster/'+str(og_name)+'.bin'
print("fname",fname, "query_idx", query_idx)
xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
rang = np.linalg.norm(xyzr[:, :3], axis=1)
range_filter = np.logical_and(rang > 0.1, rang < 80)
xyzr = xyzr[range_filter]
print("xyzr ", xyzr)

plt.figure()
plt.scatter(xyzr[:, 0], xyzr[:, 1], c=xyzr[:, 2])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("nuage de point "+str(query_idx))
plt.savefig('evaluation/image/nuage de point '+str(query_idx)+'.png')


# Get model
model = get_pipeline(cfg.eval_pipeline)

# Get checkpoint, epoch and loss
save_path = os.path.join(os.path.dirname(__file__), '../', 'checkpoints')
save_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/resultat/" 
save_path = str(save_path) + cfg.checkpoint_name
print('Loading checkpoint from: ', save_path)
checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# model in evaluation mode : normalisation layers use running statistics, de-activates Dropout layers
model = model.cuda()
model.eval()

#############################################
#tests
#############################################

if tremblement: 
    index_evolution = []
    for i in range (1, 100):
        i = i/1000
        #Création list des IDs
        list_idx = list(range(len(db_seen_descriptors))) 
         
        xyzbruit = test_bruit(xyzr, i) 
        
        if i == 0.01:
            plt.figure()
            plt.scatter(xyzbruit[:, 0], xyzbruit[:, 1], c=xyzbruit[:, 2])
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("nuage de point "+str(query_idx))
            plt.savefig('evaluation/image/nuage de point '+str(query_idx)+" n_sigma "+str(i)+'.png')
           
        input = make_sparse_tensor(xyzbruit, cfg.voxel_size).cuda()

        output_desc, output_feats = model(input)  # .squeeze()

        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        db_seen_descriptors2 = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
        feat_dists = cdist(global_descriptor, db_seen_descriptors2,
                   metric=cfg.eval_feature_distance).reshape(-1)

        feat_dists_copy = list(feat_dists)
        feat_dists_copy, list_idx = zip(*sorted(zip(feat_dists_copy, list_idx)))
        
        index_evolution.append(list_idx.index(TP_idx))
        
        
    plt.figure()
    plt.plot(index_evolution)
    plt.xlabel("n_sigma ")
    plt.ylabel("Index du TP")
    plt.title("Indice du premier TP après bruit")
    plt.savefig('evaluation/image/bruit_evol_'+str(cfg.eval_dataset)+'_query_'+str(query_idx)+'_TP_idx_'+str(TP_idx)+'.png')



elif rotation:
    index_evolution = []
    for i in range (360):
        #Création list des IDs
        list_idx = list(range(len(db_seen_descriptors))) 
        """
        if 'MulRan' in cfg.eval_dataset:
            list_file_name = []
            for i in range (len(list_idx[:5])):
                list_file_name.append( name_dict[list_idx[i]] )
            print("list_file_name ",list_file_name)
        """
        
        xyzrot = test_rotation(xyzr, i) 
        
        if i == 45:
            plt.figure()
            plt.scatter(xyzrot[:, 0], xyzrot[:, 1], c=xyzrot[:, 2])
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("nuage de point "+str(query_idx))
            plt.savefig('evaluation/image/nuage de point '+str(query_idx)+" rotation "+str(i)+'.png')
           
        input = make_sparse_tensor(xyzrot, cfg.voxel_size).cuda()

        output_desc, output_feats = model(input)  # .squeeze()

        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        db_seen_descriptors2 = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
        feat_dists = cdist(global_descriptor, db_seen_descriptors2,
                   metric=cfg.eval_feature_distance).reshape(-1)

        feat_dists_copy = list(feat_dists)
        feat_dists_copy, list_idx = zip(*sorted(zip(feat_dists_copy, list_idx)))
        
        index_evolution.append(list_idx.index(TP_idx))
        
        
    plt.figure()
    plt.plot(index_evolution)
    plt.xlabel("Rotation autour de Z")
    plt.ylabel("Index du TP")
    plt.title("Indice du premier TP après rotation")
    plt.savefig('evaluation/image/rot_evol_'+str(cfg.eval_dataset)+'_query_'+str(query_idx)+'_TP_idx_'+str(TP_idx)+'.png')
    
    
elif occlusion:
    print("occultation")
    index_evolution = []
    for i in range (360):
        #Création list des IDs
        list_idx = list(range(len(db_seen_descriptors))) 

        xyzocc = occlude_scan(xyzr, i)
        
        if i == 120:
            plt.figure()
            plt.scatter(xyzocc[:, 0], xyzocc[:, 1], c=xyzocc[:, 2])
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("nuage de point "+str(query_idx))
            plt.savefig('evaluation/image/nuage de point '+str(query_idx)+" occultation "+str(i)+'.png')
           
        input = make_sparse_tensor(xyzocc, cfg.voxel_size).cuda()

        output_desc, output_feats = model(input)  # .squeeze()

        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        db_seen_descriptors2 = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
        feat_dists = cdist(global_descriptor, db_seen_descriptors2,
                   metric=cfg.eval_feature_distance).reshape(-1)

        feat_dists_copy = list(feat_dists)
        feat_dists_copy, list_idx = zip(*sorted(zip(feat_dists_copy, list_idx)))
        index_evolution.append(list_idx.index(TP_idx))
        
    plt.figure()
    plt.plot(index_evolution)
    plt.xlabel("occultation angulaire ")
    plt.ylabel("Index du TP")
    plt.title("Indice du premier TP après occultation")
    plt.savefig('evaluation/image/occ_evol_'+str(cfg.eval_dataset)+'_query_'+str(query_idx)+'_TP_idx_'+str(TP_idx)+'.png')
    
    
    
    
    
    
    
    
    
    
    
    
a = 1
if a == 2:
    # Sorting
    feat_dists_copy = list(feat_dists)
    print("feat_dists_copy[:5] ", feat_dists_copy[:5], type(feat_dists_copy))
    print("list_idx[:5] ",list_idx[:5])
    feat_dists_copy, list_idx = zip(*sorted(zip(feat_dists_copy, list_idx)))
    print("après tri")
    print("feat_dists_copy[:5] ", feat_dists_copy[:5], type(feat_dists_copy))
    print("list_idx[:5] ",list_idx[:5])

    print(list_idx.index(TP_idx))

    # Draw min
    print('indice '+str(TP_idx), list_idx.index(TP_idx))

    plt.figure()
    plt.plot(feat_dists_copy)
    plt.axvline(x = list_idx.index(TP_idx) )

    if 'Kitti' in cfg.eval_dataset:
        plt.title('query_'+str(query_idx)+', closest TP : '+str(TP_idx)+' at '+str(list_idx.index(TP_idx)) + 'TP_dataset_'+str(cfg.eval_dataset)+'_seq_'+str(eval_seq)  )
        plt.xlabel('KNN')
        plt.ylabel('Distance')
        plt.savefig('evaluation/image/naif_'+str(cfg.eval_dataset)+'_query_'+str(query_idx)+'_TP_idx_'+str(TP_idx)+'_seq_'+str(eval_seq)+'.png')

    elif 'MulRan' in cfg.eval_dataset:
        plt.title('query_'+str(query_idx)+'_closest TP_'+str(TP_idx)+' at '+str(list_idx.index(TP_idx)) + 'TP_dat_'+str(cfg.eval_dataset)+'_seq_'+str(eval_seq)  )
        plt.suptitle('query_'+str(name_dict[query_idx])+'_closest_'+str(name_dict[TP_idx]))
        plt.xlabel('KNN')
        plt.ylabel('Distance')
        plt.savefig('evaluation/image/naif_'+str(cfg.eval_dataset)+'_query_'+str(query_idx)+'_TP_idx_'+str(TP_idx)+'.png')

    plt.close()










