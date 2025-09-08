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



def get_gbl_dsc(model, save_path, cfg, query_idx=289):
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
    print("test_loader ", test_loader)
    iterator = test_loader.__iter__()
    print("iterator", iterator)
    
    # Databases of previously visited/'seen' places.
    input_data = next(iterator)
    lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
    input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()

    output_desc, output_feats = model(input)  # .squeeze()
    output_feats = output_feats[0]

    global_descriptor = output_desc.cpu().detach().numpy()
    global_descriptor = np.reshape(global_descriptor, (1, -1))
    
    return global_descriptor


if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config import get_config_eval

    #config
    cfg = get_config_eval()
    #cfg.eval_dataset = 'MulRanDataset'

    # Get model
    model = get_pipeline(cfg.eval_pipeline)

    save_path = os.path.join(os.path.dirname(__file__), '../', 'checkpoints')
    save_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/resultat/" 
    #cfg.checkpoint_name = '/logg_epoc_31_mulran'
    save_path = str(save_path) + cfg.checkpoint_name
    print('Loading checkpoint from: ', save_path)
    
    #recherche des voisins
    query_idx = 1299
    global_descriptor = get_gbl_dsc(model, save_path, cfg, query_idx)
    
    # get seen descriptor
    if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq
        eval_seq = '%02d' % eval_seq
        
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        
    seq = str(eval_seq)

    descriptor_path = "/gpfswork/rech/dki/ujo91el/code/logg3dnet/evaluation/"+seq+"/logg3d_descriptor.pickle"
    input_file = open(descriptor_path, "rb")
    
    seen_descriptors = pickle.load(input_file)
    db_seen_descriptors = np.copy(seen_descriptors)
    #db_seen_descriptors = db_seen_descriptors[:tt+1]
    db_seen_descriptors = db_seen_descriptors.reshape(-1, np.shape(global_descriptor)[1])
    
    nearest_idx = 0
    min_dist = math.inf
    
    feat_dists = cdist(global_descriptor, db_seen_descriptors, metric=cfg.eval_feature_distance).reshape(-1)
    min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
    
    print("feat_dists", type(feat_dists), feat_dists)
    
    global_descriptor_nearest = get_gbl_dsc(model, save_path, cfg, query_idx)
    print("nearest_idx", nearest_idx)
    #place_candidate = seen_poses[nearest_idx]
    #p_dist = np.linalg.norm(query_pose - place_candidate)

    
    dif = abs(global_descriptor_nearest-global_descriptor)
    
    #print
    logging.info(
        '\n' + '******************* nearest_idx *******************')
    logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
    if 'Kitti' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.kitti_eval_seq) + ', query_idx: '+str(query_idx))
    elif 'MulRan' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.mulran_eval_seq))
    logging.info('nearest_idx: ' + str(nearest_idx))

    logging.info('global_descriptor_nearest shape: ' + str(np.shape(global_descriptor_nearest)))
    logging.info('global_descriptor_nearest type: ' + str(type(global_descriptor_nearest)))
    logging.info('global_descriptor_nearest: ' + str(global_descriptor_nearest))
    
    
    #représentation des descripteur en image
    from PIL import Image
    
    im_global_descriptor = list(global_descriptor)
    for i in range (len(global_descriptor_nearest[0])-1):
        im_global_descriptor += list(global_descriptor)
    im_global_descriptor = np.array(im_global_descriptor)
    img = Image.fromarray(np.uint8(im_global_descriptor.reshape(256,256)*255), "L")

    image_filename = "global_descriptor.png"
    img.save(image_filename)
    
    im_global_descriptor_nearest = list(global_descriptor_nearest)
    for i in range (len(global_descriptor_nearest[0])-1):
        im_global_descriptor_nearest += list(global_descriptor_nearest)
    im_global_descriptor_nearest = np.array(im_global_descriptor_nearest)
    img = Image.fromarray(np.uint8(im_global_descriptor_nearest.reshape(256,256)*255), "L")

    image_filename = "global_descriptor_nearest.png"
    img.save(image_filename)

    list_feat_dists = list(feat_dists)
    print('plot ok')
    feat_dists.sort()
    plt.figure()
    plt.plot(feat_dists)
    plt.title('Distance distribution, query:'+str(query_idx)+', nearest,' + str(nearest_idx) )
    plt.xlabel('KNN')
    plt.ylabel('Distance')
    plt.savefig('figure_'+str(query_idx)+'.png')