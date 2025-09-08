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


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, 'ab')
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.info(f'Finished saving: {file_name}')



if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config import get_config_eval


    cfg = get_config_eval()
    #cfg.eval_dataset = 'MulRanDataset'
    cfg.use_random_rotation = False
    cfg.use_random_occlusion = False
    cfg.checkpoint_name = '/logg_epoc_10_kit_check'
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(cfg.revisit_criteria), int(cfg.skip_time))
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

    #db_seen_descriptors = db_seen_descriptors[:tt+1]
    num_queries = len(positions_database)
    seen_poses, seen_descriptors, seen_feats = [], [], []
    start_time = timestamps[0]
    
    save_descriptors = True
    
    for query_idx in range (num_queries):
        
        input_data = next(iterator)
        logging.info(f'Cloud id: {query_idx}')
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
  
        output_desc, output_feats = model(input)  # .squeeze()

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
        
        if save_descriptors:
            feats = output_feats.cpu().detach().numpy()
            seen_feats.append(feats)
            continue

    if save_descriptors:
        save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        desc_file_name = '/logg3d_descriptor_notf.pickle'
        save_pickle(seen_descriptors, save_dir + desc_file_name)
        feat_file_name = '/logg3d_feats_notf.pickle'
        save_pickle(seen_feats, save_dir + feat_file_name)

