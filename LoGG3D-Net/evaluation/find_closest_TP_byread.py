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
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.mulran.mulran_dataset import load_poses_from_csv, load_timestamps_csv
from utils.data_loaders.kitti.kitti_dataset import load_poses_from_txt, load_timestamps

from models.pipeline_factory import get_pipeline
from config.eval_config import get_config_eval


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")


def main(descriptor_path, TP_file, derive_step, name_dict = None):
    #############################################
    #dataloading
    #############################################
    print('cfg.eval_dataset',cfg.eval_dataset)
    
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
    
    #data loading
    test_loader = make_data_loader(cfg,
                               cfg.test_phase,
                               cfg.eval_batch_size,
                               num_workers=cfg.test_num_workers,
                               shuffle=False)
    iterator = test_loader.__iter__()

    #############################################
    #tests
    #############################################
    
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')
        timestamps = load_timestamps(sequence_path + 'times.txt')

    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')
 
       
    
    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))

    test_loader = make_data_loader(cfg,
                                   cfg.test_phase,
                                   cfg.eval_batch_size,
                                   num_workers=cfg.test_num_workers,
                                   shuffle=False)
    
    iterator = test_loader.__iter__()
    logging.info(f'len_dataloader {len(test_loader.dataset)}')
    
    num_queries = len(positions_database)
    num_thresholds = len(thresholds)

    
    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    start_time = timestamps[0]
    
    # minimization variables
    deriv_min = np.inf
    query_idx_min = 0
    feat_dists_copy_min = 0
    TP_idx_min= 0
    list_indices_min = []
    dict_TP = {}

    # get seen global descriptor
    input_file = open(descriptor_path, "rb")
    seen_descriptors = pickle.load(input_file)
    db_seen_descriptors = np.copy(seen_descriptors)
    
    #Création list des IDs
    #list_idx = list(range(len(db_seen_descriptors))) 

    dict_TP = pickle.load(TP_file)
    print("type",type(dict_TP))   
    print("shape",np.shape(dict_TP)) 
    print("len(dict_TP_idx_min)",len(dict_TP))


    for query_idx, TP_idx in dict_TP.items():

        input_data = next(iterator)
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()

        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        
        ###################
        
        
        
        if 'Kitti' in cfg.eval_dataset:
            fname = cfg.kitti_dir + 'sequences/'+seq+'/velodyne/'+'%06d' % query_idx + '.bin'

        elif 'MulRan' in cfg.eval_dataset:
            og_name = name_dict[query_idx]
            fname = '/gpfswork/rech/dki/ujo91el/datas/mulran/'+seq+'/Ouster/'+str(og_name)+'.bin'
            print("fname",fname, og_name, "query_idx", query_idx, )
            
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        rang = np.linalg.norm(xyzr[:, :3], axis=1)
        range_filter = np.logical_and(rang > 0.1, rang < 80)
        xyzr = xyzr[range_filter]
        input = make_sparse_tensor(xyzr, cfg.voxel_size).cuda()
        
        #####################
        
        output_desc, output_feats = model(input)  # .squeeze()
    
        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        
        query_time = timestamps[query_idx]
        
        if len(global_descriptor) < 1:
            continue

        if (query_time - start_time - cfg.skip_time) < 0:
            continue


        # Find top-1 candidate.

        db_seen_descriptors2 = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
        feat_dists = cdist(global_descriptor, db_seen_descriptors2,
                           metric=cfg.eval_feature_distance).reshape(-1)

        # Sorting
        feat_dists_copy = list(feat_dists)

        try:
            derive = (feat_dists_copy[derive_step]-feat_dists_copy[0])/derive_step
        except :
            derive = np.inf
            
        #Save slowest growth
        if derive < deriv_min:
            print("query_idx", query_idx)
            print("new min", derive)
            deriv_min = derive
            query_idx_min = query_idx
            feat_dists_copy_min = feat_dists_copy
            list_idx = list(range(len(feat_dists)))
            TP_idx_min= TP_idx 
   

    print("feat_dists_copy_min[:5] ", feat_dists_copy_min[:5], type(feat_dists_copy_min))
    print("list_idx_min[:5] ",list_idx[:5])
    feat_dists_copy_min, list_idx = zip(*sorted(zip(feat_dists_copy_min, list_idx)))
    print("après tri")
    print("feat_dists_copy_min[:5] ", feat_dists_copy_min[:5], type(feat_dists_copy_min))
    print("list_idx_min[:5] ",list_idx[:5])

    # Draw min
    print("query ", query_idx_min, ' indice '+str(TP_idx_min), list_idx.index(TP_idx_min))
    print('plot ok ', abs(deriv_min))

    plt.figure()
    plt.plot(feat_dists_copy_min)
    plt.axvline(x = list_idx.index(TP_idx_min))
    
    if 'Kitti' in cfg.eval_dataset:
        plt.title('query_'+str(query_idx_min)+'_TP_idx'+str(TP_idx_min)+' at '+str(list_idx.index(TP_idx_min)) +'_TP_dat_'+str(cfg.eval_dataset)+'_seq_'+str(eval_seq))
        plt.xlabel('KNN')
        plt.ylabel('Distance')
        plt.savefig('evaluation/image/'+str(cfg.eval_dataset)+'_query_'+str(query_idx_min)+'_TP_idx_'+str(TP_idx_min)+'_seq_'+str(eval_seq)+'.png')
   
    elif 'MulRan' in cfg.eval_dataset:    
        plt.title('query_'+str(query_idx_min)+'_TP_idx_'+str(TP_idx_min)+' at '+str(list_idx.index(TP_idx_min)) + '_TP_dat'+str(cfg.eval_dataset)+'_seq_'+str(eval_seq))
        plt.suptitle('query_'+str(name_dict[query_idx_min])+'_TP_idx_'+str(name_dict[TP_idx_min]))
        plt.xlabel('KNN')
        plt.ylabel('Distance')
        plt.savefig('evaluation/image/'+str(cfg.eval_dataset)+'_query_'+str(query_idx_min)+'_TP_idx_'+str(TP_idx_min)+'.png')
    
    plt.close()
    return 

if __name__ == "__main__":
    cfg = get_config_eval()

    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        eval_seq = '%02d' % eval_seq
        name_dict = None
        
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        #check name for mulran
        name_file = open("/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/classification/"+str(cfg.mulran_eval_seq)+"_Names_MulRanDataset.pkl", "rb")
        name_dict = pickle.load(name_file)
        
        print("name_dict[3671]",name_dict[3671])
        print("name_dict[3468]",name_dict[3468])

        
    seq = str(eval_seq)
    dat = str(cfg.eval_dataset)
    
    
    descriptor_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/"+seq+"/logg3d_descriptor.pickle"
    print("descriptor_path", descriptor_path)
    
    TP_file = open("/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/classification/"+seq+'_seq_TP_dataset_'+dat+'.pkl', "rb")
    print("input_file", TP_file)
    
    derive_step = 200
    main(descriptor_path, TP_file, derive_step, name_dict )    
