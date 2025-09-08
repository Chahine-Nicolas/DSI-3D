import os
import sys
import torch
import logging
from scipy.spatial.distance import cdist
import logging
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



ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")



if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config import get_config_eval

    cfg = get_config_eval()
    #cfg.eval_dataset = 'MulRanDataset'

    # Get model
    model = get_pipeline(cfg.eval_pipeline)

    save_path = os.path.join(os.path.dirname(__file__), '../', 'checkpoints')
    save_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/resultat/" 
    #cfg.checkpoint_name = '/logg_epoc_31_mulran'
    save_path = str(save_path) + cfg.checkpoint_name
    print('Loading checkpoint from: ', save_path)
    
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    model.eval()
    test_loader = make_data_loader(cfg,
                               cfg.test_phase,
                               cfg.eval_batch_size,
                               num_workers=cfg.test_num_workers,
                               shuffle=False)
    iterator = test_loader.__iter__()
    
    

    
    print('cfg.eval_dataset',cfg.eval_dataset)
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')

    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
    

    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))
    
    num_thresholds = len(thresholds)
    
    
    # get seen descriptor
    input_file = open("/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/06/logg3d_descriptor.pickle", "rb")
    seen_descriptors = pickle.load(input_file)
    db_seen_descriptors = np.copy(seen_descriptors)
    #db_seen_descriptors = db_seen_descriptors[:tt+1]
   
    num_queries = len(positions_database)
    
    deriv_min = np.inf
    query_min = 0
    fdits = []
    features = []
    min_q = []
    
    seen_poses = []
    for query_idx in range (num_queries):
        query_pose = positions_database[query_idx]
        seen_poses.append(query_pose)
        
        
    for query_idx in range (num_queries):
        print('query', query_idx, ' / ', num_queries)
        
        input_data = next(iterator)
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()

        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        output_desc, output_feats = model(input)  # .squeeze()

        output_feats = output_feats[0]

        global_descriptor = output_desc.cpu().detach().numpy()
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        

        db_seen_descriptors2 = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
        feat_dists = cdist(global_descriptor, db_seen_descriptors2,
                           metric=cfg.eval_feature_distance).reshape(-1)
        feat_copy = feat_dists.tolist()
        
        list_query = list(range(len(feat_copy)))
        feat_copy, list_query = zip(*sorted(zip(feat_copy, list_query)))


        derive = (feat_copy[200]-feat_copy[0])/200
        
        #enregistre le cas le plus défavorable
        if derive < deriv_min:
            print("new min", derive)
            deriv_min = derive
            query_min = query_idx
            
            features = feat_copy
            min_q = list_query
            fdits = feat_dists
    
    print("plus petite dervivee", derive)       
    print("query_min = plus proch dans l'espace latent ", query_min)
    print("features distances triee", features)
    print("list des query ", min_q)
    print("features distances non trie ", fdits)
    
    print("min_dist", np.min(fdits), "np.argmin(fdits)", np.argmin(fdits))
    min_dist, nearest_idx = np.min(fdits), np.argmin(fdits)
    place_candidate = seen_poses[nearest_idx]
    p_dist = np.linalg.norm(query_pose - place_candidate)

    #trouver l'id du nuage TP   
    for thres_idx in range(num_thresholds):
        threshold = thresholds[thres_idx]

        if(min_dist < threshold):  # Positive Prediction
            if p_dist <= cfg.revisit_criteria:
                print("nearest id:"+str(nearest_idx))

    #draw min
    nearest_idx = 260
    print('indice '+str(nearest_idx),', index ', min_q.index(nearest_idx))
    print('plot ok ', abs(deriv_min))
    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    logging.info(f'cfg.eval_dataset {cfg.eval_dataset} cfg.checkpoint_name  {cfg.checkpoint_name }')
    plt.figure()
    plt.plot(features)
    print("axe", min_q.index(nearest_idx))
    plt.axvline(x = min_q.index(nearest_idx) )
    plt.title('Distance distribution, query:'+str(query_min))
    plt.xlabel('KNN')
    plt.ylabel('Distance')
    plt.savefig('image/figure_'+str(query_min)+'.png')
    plt.close()




