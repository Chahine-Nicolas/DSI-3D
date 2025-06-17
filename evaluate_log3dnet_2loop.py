from scipy.spatial.distance import cdist
import os
import sys
import glob
import random
import numpy as np
import logging
import json
import torch
import math
#from pathlib import Path
import matplotlib.pyplot as plt
#####################################################################################
# Load poses
# ####################################################################################
import time

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

def _pad_tensors_to_max_len( tensor, max_length,tokenizer):
    # If PAD token is not defined at least EOS token has to be defined
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    tensor[tensor == -100] = tokenizer.pad_token_id
    padded_tensor = pad_token_id * torch.ones(
        (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor


def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo


def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
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
# ####################################################################################


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

#####################################################################################
# Timing


class Timer(object):
    """A simple timer."""
    # Ref: https://github.com/chrischoy/FCGF/blob/master/lib/timer.py

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0

    @property
    def avg(self):
        return self.total_time / self.calls

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff

""
def eval_log3dnet(model,eval_subset,eval_set,data_collator,tokenizer,cfg,checkpoint_dir, do_eval_partial):
    print("eval lognet")

    #eval_set.labeltype = 'log3dnet'
    save_descriptors = False
    save_counts = False
    plot_pr_curve = True
    
    eval_seq=cfg['DATA_CONFIG']['SEQ']
    log3dnet_dir=os.getenv('LOG3DNET_DIR')
    revisit_criteria=3
    not_revisit_criteria=20
    skip_time=30
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(revisit_criteria), int(skip_time))
    cd_thresh_min=0.001
    cd_thresh_max=5 # au lieu de 1
    num_thresholds=1000
    num_beams = 10
    ## ==== Kitti =====
    print("kitti dataset")
    kitti_dir= os.getenv('WORKSF') + '/datas/datasets/'
    sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
    _, positions_database = load_poses_from_txt(sequence_path + 'poses.txt')

    min_bbox = np.min(positions_database,0) 
    positions_database = positions_database - min_bbox
    
    timestamps = load_timestamps(sequence_path + 'times.txt')
    revisit_json_dir = os.path.join(os.path.dirname(__file__), '/config/kitti_tuples/')
    revisit_json = json.load(open(log3dnet_dir + revisit_json_dir + revisit_json_file, "r"))
    is_revisit_list = revisit_json[eval_seq]


    # import pdb; pdb.set_trace()        
    ## ==== Kitti =====

    thresholds = np.linspace(
        cd_thresh_min, cd_thresh_max, int(num_thresholds))

    
    num_queries = len(positions_database)
    num_queries = len(eval_subset)
    num_thresholds = len(thresholds)

    # Databases of previously visited/'seen' places.
    seen_poses, seen_ids, seen_descriptors, seen_feats = [], [], [], []

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)

    name_true_positive = [ [] for i in range(int(num_thresholds)) ]
    name_false_positive = [ [] for i in range(int(num_thresholds)) ]
    name_true_negative = [ [] for i in range(int(num_thresholds)) ]
    name_false_negative = [ [] for i in range(int(num_thresholds)) ]

    num_true_positive_beams = np.zeros([num_beams,num_thresholds])
    num_false_positive_beams = np.zeros([num_beams,num_thresholds])
    num_true_negative_beams = np.zeros([num_beams,num_thresholds])
    num_false_negative_beams = np.zeros([num_beams,num_thresholds])

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    num_correct_loc_beam = np.zeros(num_beams)
    start_time = timestamps[0]
    dictio = []
    print("Start looop")

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()
    

    #import pdb; pdb.set_trace()

    #model.from_pretrained("/lustre/fswork/projects/rech/xhk/ufm44cu/out/dsi_iter_256_v2/1536/")
    model_path = checkpoint_dir +  "/1_1/pytorch_model.bin"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, False)
    model.eval()
    print("model_path:" + model_path)


    
    for query_idx in range(num_queries):
        #if query_idx < 621 and do_eval_partial :
            #continue
        if query_idx > 1700 and do_eval_partial :
            break
            
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]

        seen_poses.append(query_pose)
        seen_ids.append('%06d' % query_idx)

        if eval_set.labeltype == 'log3dnet' or True:
            desc_path = sequence_path + 'logg_desc/' +  ('%06d' % query_idx) + '.pt'
            global_descriptor = torch.load(desc_path).detach().cpu().numpy()
            global_descriptor = np.reshape(global_descriptor, (1, -1))
            seen_descriptors.append(global_descriptor)


        #skip_time = 0
        if (query_time - start_time - skip_time) < 0:
            continue

        # import pdb; pdb.set_trace()
        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - skip_time))

        db_seen_poses = np.copy(seen_poses)
        db_seen_poses = db_seen_poses[:tt+1]
        db_seen_ids = np.copy(seen_ids)
        db_seen_ids = db_seen_ids[:tt+1]

        is_revisit = is_revisit_list[query_idx]
        

        ### Restrict decod vocab
        LIK = []
        ID_MAX_LENGTH=10
        if eval_set.labeltype == 'gps' :
            for ii in db_seen_ids : LIK.append(tokenizer(eval_set.label2gps(ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids) 
        elif eval_set.labeltype == 'hierarchical' :
            for ii in db_seen_ids : LIK.append(tokenizer(eval_set.get_hierarchical_label(ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        else :
            for ii in db_seen_ids : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        #import pdb; pdb.set_trace()        
        
        def restrict_decode_vocab(batch_idx, prefix_beam):
            TOK_ID_OK = []
            sz = len(prefix_beam)
            pfb = prefix_beam.cpu().numpy()
            for kk in LIK :
                if kk[:sz] == pfb.tolist()  :
                    TOK_ID_OK.append(kk[sz])
            if len(TOK_ID_OK) == 0 :
                TOK_ID_OK.append(102)
            return TOK_ID_OK


        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        #ret_timer.tic()
        ## HERE CALL DSI
        ## Modif
        
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])


        beam_ids = None
        if eval_set.labeltype != 'log3dnet' :
            
            if ((query_idx-1)%256==0 and query_idx != 1 ) and (not do_eval_partial) :
                if 256<(query_idx)<512: 
                     model_path = checkpoint_dir +  "/1_1/pytorch_model.bin"
                elif 512<(query_idx)<768: 
                     model_path = checkpoint_dir +  "/2_2/pytorch_model.bin"
                elif 768<(query_idx)<1024: 
                     model_path = checkpoint_dir +  "/3_3/pytorch_model.bin"

                print("model_path:" + model_path)
                if os.path.exists(model_path) :
                    print("LOAD NEW PATH:" + model_path) 
                    state_dict = torch.load(model_path)
                    model.load_state_dict(state_dict, False)
                    model.eval()
                else :
                    print("WARNIGN!! checkoint does not exists, skip, eval will be wrong")

            # if is_revisit == 0.0 :
            #     print("not revisit! continue for quick test!" + str(query_idx))
            #     continue
            prep_timer.tic()   
            input_data = data_collator(torch.utils.data.Subset(eval_subset,range(query_idx, query_idx+1)))            #model_id = int((int(db_seen_ids[-1])//256+1)*256)
            prep_timer.toc()
            desc_timer.tic()
            with torch.no_grad():
                batch_beams_dict = model.generate(
                        #pixel_values=None,
                        pixel_values=input_data['pixel_values'],
                        lidar_values=input_data['lidar_values'],
                        points=None,
                        #points=inputs['lidar_values']['points'],
                        max_length=ID_MAX_LENGTH,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        eos_token_id=102,
                        pad_token_id=0,
                        bos_token_id=101,
                        renormalize_logits=False,
                        early_stopping=False, #True,#
                        prefix_allowed_tokens_fn=restrict_decode_vocab,
                        return_dict_in_generate=True,                
                        output_scores = True,
                        )
            desc_timer.toc()
            ret_timer.tic()
            batch_beams = batch_beams_dict['sequences']
            seq_score = batch_beams_dict['sequences_scores'].reshape([-1, num_beams])
            res = _pad_tensors_to_max_len(input_data['labels'], ID_MAX_LENGTH,tokenizer)


            vv = tokenizer.batch_decode(input_data["labels"],skip_special_tokens=True)
            ids = input_data['ids']

            label_ids = tokenizer.batch_decode(batch_beams, skip_special_tokens=True)
            ret_timer.toc()
            #print("label_ids ", label_ids)
            #import pdb; pdb.set_trace()
            p_dist_mean = 0
            p_dist_beam=0
            beam_ids = None
            
            if eval_set.labeltype == 'gps' : #eval_set.gpslabel :

                #nearest_idx = label_ids[0]
                #place_candidate = eval_set.gps2position(nearest_idx)

                ######################################################################################################################
                # convert GPS to label
                ######################################################################################################################
                
                f = open(sequence_path+"dict_gps_2_label_v2.json",) # gps xyxyxyxy
                data = json.load(f)
                f.close()
                #label_ids = [ data[x] for x in label_ids]
                #all_idx = ' '.join(label_ids)
                ######################################################################################################################
                
                #if nearest_idx == '206776787':
                    #nearest_idx = '2067767873'
                nearest_idx = data[label_ids[0]]
                place_candidate = seen_poses[int(nearest_idx)]
              
                #import pdb; pdb.set_trace()
                place_candidate_mean = np.mean([seen_poses[int(data[kk])][:2] for kk in label_ids],0)
                p_dist = np.linalg.norm(query_pose[:2] - place_candidate[:2])
                p_dist_mean = np.linalg.norm(query_pose[:2] - place_candidate_mean[:2])
                p_dist_beam = min([np.linalg.norm(query_pose[:2] - seen_poses[int(data[kk])][:2]) for kk in label_ids])

                min_dist_beam = 1 - np.exp(seq_score[0][0].cpu().numpy())
                
                
            elif  eval_set.labeltype == 'hierarchical' : #eval_set.gpslabel :
                nearest_idx = int(eval_set.inv_hierarchical_label[label_ids[0]])
                place_candidate = seen_poses[nearest_idx]
                p_dist_beam = min([np.linalg.norm(query_pose -seen_poses[int(eval_set.inv_hierarchical_label[kk])]) for kk in label_ids])
                p_dist = np.linalg.norm(query_pose - place_candidate)
                p_dist_mean = 0


            else : 
                nearest_idx = label_ids[0]
                place_candidate = seen_poses[int(nearest_idx)]
                p_dist = np.linalg.norm(query_pose - place_candidate)
                #import pdb; pdb.set_trace()        
                # beam_ids = [int(kk) for kk in label_ids]
                
            beam_ids = None
            if beam_ids is not None :
                min_dist_beam = np.zeros(num_beams)
                p_dist_beam = np.zeros(num_beams)
                for bb in range(0,num_beams) :
                    db_seen_descriptors_beam = np.copy(seen_descriptors)
                    db_seen_descriptors_beam = db_seen_descriptors_beam[beam_ids[:(bb+1)]]
                    db_seen_descriptors_beam = db_seen_descriptors_beam.reshape(
                        -1, np.shape(global_descriptor)[1])
                    feat_dists_beam = cdist(global_descriptor, db_seen_descriptors_beam,
                                            metric='cosine').reshape(-1)
                    argmm = np.argmin(feat_dists_beam)
                    place_candidate_beam = seen_poses[beam_ids[argmm]]
                    min_dist_beam[bb] = feat_dists_beam[argmm]
                    p_dist_beam[bb] = np.linalg.norm(query_pose - place_candidate_beam)

                    #nearest_idx_beam = sorted(range(len(feat_dists_beam)), key=lambda i: feat_dists_beam[i])
                    # place_candidate_beam_full = np.asarray(seen_poses)[np.asarray(beam_ids[:bb])[nearest_idx_beam]]
                    # p_dist_beam_full = np.linalg.norm(np.tile(query_pose, (bb, 1)) - place_candidate_beam_full,2,1)
                    #import pdb; pdb.set_trace()        
                
            
        
            #min_dist = (1-np.exp(seq_score[0][0].cpu().numpy()))   # change here for seuil definition
            min_dist = -seq_score[0][0].cpu().numpy()
        else :
            feat_dists = cdist(global_descriptor, db_seen_descriptors,
                               metric='cosine').reshape(-1)
            min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
            # ret_timer.toc()
            place_candidate = seen_poses[nearest_idx]
            p_dist = np.linalg.norm(query_pose - place_candidate)

        # is_revisit = check_if_revisit(query_pose, db_seen_poses, cfg.revisit_criteria)            
        is_revisit = is_revisit_list[query_idx]
        is_correct_loc = 0
        is_correct_loc_beam = np.zeros(num_beams)
        #
        if is_revisit:
            #import pdb; pdb.set_trace()        
            num_revisits += 1
            if p_dist <= revisit_criteria:
                num_correct_loc += 1
                is_correct_loc = 1
            if beam_ids is not None : 
                for bb in range(0,num_beams) :
                    if p_dist_beam[bb] <= revisit_criteria:
                        num_correct_loc_beam[bb] += 1
                        is_correct_loc_beam[bb] = 1                

        #logging.info(f'id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} [loc_ok_1:{is_correct_loc} min_d_1:{min_dist:.2f} p_dist_1:{p_dist:6.2f} p_dist_mean:{p_dist_mean},  loc_ok_beam:{is_correct_loc_beam} p_dist_beam:{p_dist_beam:6.2f}')
        #logging.info(f'id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} [loc_ok_1:{is_correct_loc} p_dist_1:{p_dist:6.2f} p_dist_mean:{p_dist_mean:6.2f},  loc_ok_beam:{is_correct_loc_beam} p_dist_beam:{p_dist_beam:6.2f}')
        if  eval_set.labeltype == 'hierarchical':
            logging.info(f'id:{query_idx} id_hierar {eval_set.hierarchical_label[str(query_idx)]} Top1_hierar:{label_ids[0]} Top1_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} ')
            
        elif  eval_set.labeltype == 'gps':
            logging.info(f'Query:{query_idx} Query_gps {eval_set.label2gps(str(query_idx))} Top1_gps:{label_ids[0]} Top1_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} ')
        
        else:
            logging.info(f'id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} ')

        #saved predictions
        dictio.append({"query_idx":query_idx,"Top1_id":nearest_idx, "is_rev":is_revisit, "loc_ok_1":is_correct_loc, "p_dist":p_dist, "min_dist":min_dist})

        if beam_ids is not None : 
            print('[beam:' + ' '.join(map(str, is_correct_loc_beam.astype(int))) + ']')
            print('[beam:' + ' '.join(f'{x:.2f}' for x in p_dist_beam) + ']')
        #import pdb; pdb.set_trace()                    
        
        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= revisit_criteria :
                    num_true_positive[thres_idx] += 1
                    #name_true_positive[thres_idx].append({query_idx:nearest_idx}) 

                elif p_dist > not_revisit_criteria:
                    num_false_positive[thres_idx] += 1
                    name_false_positive[thres_idx].append([f'id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} '])  
                    
            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                    #name_true_negative[thres_idx].append({query_idx:nearest_idx}) 
                    
                else:
                    num_false_negative[thres_idx] += 1
                    #name_false_negative[thres_idx].append({query_idx:nearest_idx}) 

        # # Evaluate top-beam candidate.
        # if beam_ids is not None : 
        #     for thres_idx in range(num_thresholds):
        #         threshold = thresholds[thres_idx]
        #         for bb in range(0,num_beams) :
        #             if(min_dist_beam[bb] < threshold):  # Positive Prediction
        #                 if p_dist_beam[bb] <= revisit_criteria :
        #                     num_true_positive_beams[bb][thres_idx] += 1

        #                 elif p_dist_beam[bb] > not_revisit_criteria:
        #                     num_false_positive_beams[bb][thres_idx] += 1

        #             else:  # Negative Prediction
        #                 if(is_revisit == 0):
        #                     num_true_negative_beams[bb][thres_idx] += 1
        #                 else:
        #                     num_false_negative_beams[bb][thres_idx] += 1                    

    
    F1max = 0.0
    Precisions, Recalls = [], []
    if not save_descriptors:

        ## Original
        for ithThres in range(num_thresholds):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres]

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)

                F1 = 2 * Precision * Recall * (1/(Precision + Recall))

            if F1 > F1max:
                F1max = F1
                F1_TN = nTrueNegative
                F1_FP = nFalsePositive
                F1_TP = nTruePositive
                F1_FN = nFalseNegative
                F1_thresh_id = ithThres
            Precisions.append(Precision)
            Recalls.append(Recall)
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(
            f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
        logging.info(
            f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(
            f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        logging.info(f'F1_thresh_id: {F1_thresh_id}')
        logging.info(f'F1max: {F1max}')


        logging.info('Average times per scan:')
        logging.info(
            f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s Ret: {ret_timer.avg}s ---")
        logging.info('Average total time per scan:')
        logging.info(
            f"--- {prep_timer.avg + desc_timer.avg + ret_timer.avg}s ---")

        
        # Beams
        # for bb in range(0,num_beams) :
        #     for ithThres in range(num_thresholds):
        #         bb_id = bb+1
        #         nTrueNegative = num_true_negative_beams[bb][ithThres]
        #         nFalsePositive = num_false_positive_beams[bb][ithThres]
        #         nTruePositive = num_true_positive_beams[bb][ithThres]
        #         nFalseNegative = num_false_negative_beams[bb][ithThres]
        #         Precision = 0.0
        #         Recall = 0.0
        #         F1 = 0.0
        #         if nTruePositive > 0.0:
        #             Precision = nTruePositive / (nTruePositive + nFalsePositive)
        #             Recall = nTruePositive / (nTruePositive + nFalseNegative)
        #             F1 = 2 * Precision * Recall * (1/(Precision + Recall))
        #         if F1 > F1max:
        #             F1max = F1
        #             F1_TN = nTrueNegative
        #             F1_FP = nFalsePositive
        #             F1_TP = nTruePositive
        #             F1_FN = nFalseNegative
        #             F1_thresh_id = ithThres
        #         Precisions.append(Precision)
        #         Recalls.append(Recall)
        #     print(f'========== MAX BEAMS : {bb_id} ===============')
        #     print(f'num_revisits: {num_revisits}')
        #     print(f'num_correct_loc: {num_correct_loc}')
        #     print(f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
        #     print(f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        #     print(f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        #     print(f'F1_thresh_id: {F1_thresh_id}')
        #     print(f'F1max: {F1max}')

        
        if eval_set.labeltype == 'log3dnet' :
            checkpoint_dir = "/lustre/fswork/projects/rech/xhk/ufm44cu/checkpoints/logg3dnet"
        
        if plot_pr_curve:
            plt.figure()
            plt.title('Seq: ' + str(eval_seq) +
                      '    F1Max: ' + "%.4f" % (F1max))
            plt.plot(Recalls, Precisions, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1, 0, 1.1])
            plt.xticks(np.arange(0, 1.01, step=0.1))
            plt.grid(True)
            save_dir = os.path.join(checkpoint_dir, 'pr_curves')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_seq = str(eval_seq).split('/')[-1]
            plt.savefig(save_dir + '/' + eval_seq + '.png')

    """
    if not save_descriptors:
        logging.info('Average times per scan:')
        logging.info(
            f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s Ret: {ret_timer.avg}s ---")
        logging.info('Average total time per scan:')
        logging.info(
            f"--- {prep_timer.avg + desc_timer.avg + ret_timer.avg}s ---")
    """
    
    if save_descriptors:
        save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        desc_file_name = '/logg3d_descriptor.pickle'
        save_pickle(seen_descriptors, save_dir + desc_file_name)
        feat_file_name = '/logg3d_feats.pickle'
        save_pickle(seen_feats, save_dir + feat_file_name)

    save_counts = False
    if save_counts:
        save_dir = os.path.join(checkpoint_dir, 'pickles/', str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
        save_pickle(num_false_positive, save_dir +
                    '/num_false_positive.pickle')
        save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
        save_pickle(num_false_negative, save_dir +
                    '/num_false_negative.pickle')
     

    # Build stat    
    list_true_positive = []
    list_false_positive = []
    list_true_negative = []
    list_false_negative = []
    
    threshold = F1_thresh_id * 0.005
    
    for query_idx in range(len(dictio)):
        if(dictio[query_idx]["min_dist"] < threshold):  # Positive Prediction
            if dictio[query_idx]["p_dist"] <= 3 :
                list_true_positive.append(dictio[query_idx])
            elif dictio[query_idx]["p_dist"] > 20:
                list_false_positive.append(dictio[query_idx])       
        else:  # Negative Prediction
            if(dictio[query_idx]["is_rev"] == 0):
                list_true_negative.append(dictio[query_idx])    
            else:
                list_false_negative.append(dictio[query_idx])

    print(len(list_true_negative),len(list_false_positive), len(list_true_positive), len(list_false_negative))

    with open(eval_seq+"_id_true_negative.json", "w") as final: json.dump([d['query_idx'] for d in list_true_negative], final)
    with open(eval_seq+"_id_false_positive.json", "w") as final: json.dump([d['query_idx'] for d in list_false_positive], final)
    with open(eval_seq+"_id_true_positive.json", "w") as final: json.dump([d['query_idx'] for d in list_true_positive], final)
    with open(eval_seq+"_id_false_negative.json", "w") as final: json.dump([d['query_idx'] for d in list_false_negative], final)

    print_class = ["TP", "FP", "FN", "TN"]
    
    for scor_key in ['p_dist', 'min_dist']: 
        count = 0
        print(scor_key)
        if scor_key == 'p_dist':
            round_num = 2
        else:
            round_num = 4
        for dict_to in [list_true_positive, list_false_positive, list_false_negative, list_true_negative]: 
            print(print_class[count])
            print("mean ", round(sum(d[str(scor_key)] for d in dict_to) / len(dict_to),round_num))
            print("min ", round(min(dict_to, key=lambda x:x[str(scor_key)])[str(scor_key)],round_num))
            print("max ", round(max(dict_to, key=lambda x:x[str(scor_key)])[str(scor_key)],round_num))
            listr = []
            for k in range(len(dict_to)): listr.append(dict_to[k][str(scor_key)])
            print("std ", round(np.std(listr),round_num))
            count += 1

    import pdb; pdb.set_trace() 
            
    """
    # p_dist
    dict_to = list_true_positive
    print(sum(d['p_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['p_dist']))
    print(max(dict_to, key=lambda x:x['p_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['p_dist'])
    print(np.std(listr))
    
    dict_to = list_false_positive
    print(sum(d['p_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['p_dist']))
    print(max(dict_to, key=lambda x:x['p_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['p_dist'])
    print(np.std(listr))

    dict_to = list_false_negative
    print(sum(d['p_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['p_dist']))
    print(max(dict_to, key=lambda x:x['p_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['p_dist'])
    print(np.std(listr))
    
    dict_to = list_true_negative
    print(sum(d['p_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['p_dist']))
    print(max(dict_to, key=lambda x:x['p_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['p_dist'])
    print(np.std(listr))

    # min_dist
    dict_to = list_true_positive
    print(sum(d['min_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['min_dist']))
    print(max(dict_to, key=lambda x:x['min_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['min_dist'])
    print(np.std(listr))
    
    dict_to = list_false_positive
    print(sum(d['min_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['min_dist']))
    print(max(dict_to, key=lambda x:x['min_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['min_dist'])
    print(np.std(listr))

    dict_to = list_false_negative
    print(sum(d['min_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['min_dist']))
    print(max(dict_to, key=lambda x:x['min_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['min_dist'])
    print(np.std(listr))
    
    dict_to = list_true_negative
    print(sum(d['min_dist'] for d in dict_to) / len(dict_to))
    print(min(dict_to, key=lambda x:x['min_dist']))
    print(max(dict_to, key=lambda x:x['min_dist']))
    listr = []
    for k in range(len(dict_to)): listr.append(dict_to[k]['min_dist'])
    print(np.std(listr))
    """
    
    
    return F1max                    
