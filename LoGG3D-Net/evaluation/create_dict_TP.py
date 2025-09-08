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

from models.pipeline_factory import get_pipeline
from config.test_config import get_config_test



ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")


def main(derive_step):
    
        #Get config
    cfg = get_config_test()
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
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(cfg.revisit_criteria), int(cfg.skip_time))
    
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        cfg.kitti_data_split['test'] = [eval_seq]
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')
        timestamps = load_timestamps(sequence_path + 'times.txt')
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/kitti_tuples/')
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]
        
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/mulran_tuples/')
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

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
    
    # Databases of previously visited/'seen' places.
    seen_poses, seen_descriptors, seen_feats = [], [], []
    
    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
    
    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()
    
    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    start_time = timestamps[0]
    
    dict_tp = {}
    dict_fp = {}
    dict_tn = {}
    dict_fn = {}

	###############################
    for query_idx in range(num_queries):
        print("query_idx", query_idx,' / ',num_queries)
        input_data = next(iterator)
        prep_timer.tic()
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
        if not len(lidar_pc) > 0:
            logging.info(f'Corrupt cloud id: {query_idx}')
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        prep_timer.toc()
        desc_timer.tic()
        output_desc, output_feats = model(input)  # .squeeze()
        desc_timer.toc()
        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()
        
        global_descriptor = np.reshape(global_descriptor, (1, -1))
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]
        
        if len(global_descriptor) < 1:
            continue

        seen_descriptors.append(global_descriptor)
        seen_poses.append(query_pose)

        if (query_time - start_time - cfg.skip_time) < 0:
            continue

        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - cfg.skip_time))
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_poses = np.copy(seen_poses)
        db_seen_poses = db_seen_poses[:tt+1]
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])

        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        ret_timer.tic()
        feat_dists = cdist(global_descriptor, db_seen_descriptors,
                           metric=cfg.eval_feature_distance).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
        ret_timer.toc()

        place_candidate = seen_poses[nearest_idx]
        p_dist = np.linalg.norm(query_pose - place_candidate)

        # is_revisit = check_if_revisit(query_pose, db_seen_poses, cfg.revisit_criteria)
        try:
            is_revisit = is_revisit_list[query_idx]
        except:
            break
        is_correct_loc = 0
        if is_revisit:
            num_revisits += 1
            if p_dist <= cfg.revisit_criteria:
                num_correct_loc += 1
                is_correct_loc = 1

        logging.info(
            f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist} p_dist: {p_dist}')

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist
            
        list_query = []

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]
            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= cfg.revisit_criteria:
                    dict_tp[query_idx] = nearest_idx

                elif p_dist > cfg.not_revisit_criteria:
                    dict_fp[query_idx] = nearest_idx

            else:  # Negative Prediction
                if(is_revisit == 0):
                    dict_tn[query_idx] = nearest_idx

                else:
                    dict_fn[query_idx] = nearest_idx
    
    #print("dict_tp", dict_tp)
    with open('evaluation/classification/'+str(eval_seq)+'_seq_TP_dataset_'+str(cfg.eval_dataset)+'.pkl', 'wb') as f:
        pickle.dump(dict_tp, f)
    return 

if __name__ == "__main__":
    derive_step = 200
    main(derive_step)