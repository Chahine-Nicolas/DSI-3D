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


def eval_log3dnet(model_expert_1, model_expert_2, model_expert_3, model_expert_4, model_expert_5, model_expert_6, eval_subset, eval_set, eval_loader, expert_data_collator, tokenizer, cfg, checkpoint_dir, expert_checkp, expert_tokenizer, expert_prefix, expert_hierar_label, expert_eval_subset, encoder_chekp=None):
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
    revisit_json_file = 'is_revisit_D-{}_T-{}_v2.json'.format(
        int(revisit_criteria), int(skip_time))
    cd_thresh_min=0.001
    cd_thresh_max=5 # au lieu de 1
    num_thresholds=5000
    num_beams = 10
    ## ==== Kitti =====
    print("kitti dataset")
    kitti_dir= os.getenv('WORKSF') + '/datas/datasets/'
    sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'

    def get_position_database(eval_seq):
        sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(sequence_path + 'poses.txt')
        min_bbox = np.min(positions_database,0) 
        positions_database = positions_database - min_bbox
        return positions_database

    positions_database = get_position_database(eval_seq) #22
    positions_database_1 = get_position_database("00") #00
    positions_database_2 = get_position_database("02") #02
    positions_database_3 = get_position_database("05") #05
    positions_database_4 = get_position_database("06") #06
    positions_database_5 = get_position_database("07") #07
    positions_database_6 = get_position_database("08") #08


    
    
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
    
    ######################################################################################
    # classification binaire sans zones grises
    ######################################################################################
    num_true_positive_3m = np.zeros(num_thresholds)
    num_false_positive_3m = np.zeros(num_thresholds)
    num_true_negative_3m = np.zeros(num_thresholds)
    num_false_negative_3m = np.zeros(num_thresholds)
    ######################################################################################
    ######################################################################################

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0
    num_correct_loc = 0
    num_correct_loc_all = 0
    hit_at_10 = 0
    dictio = []
    print("Start looop")

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()


    
    ###########################
    # load expert chkpt

    def load_chkp(model, checkpoint_dir ,checkp_to_eval):
        chkp = checkpoint_dir +  "/" + checkp_to_eval + "/pytorch_model.bin" 
        print("load "+ chkp)
        state_dict = torch.load(chkp)

        model.load_state_dict(state_dict, False)
        model.eval()
        
        return model

    """
    model_expert_1 = load_chkp(model_expert_1, checkpoint_dir[0] ,checkp_to_eval[0]) #00
    model_expert_2 = load_chkp(model_expert_2, checkpoint_dir[1] ,checkp_to_eval[1]) #02
    model_expert_3 = load_chkp(model_expert_3, checkpoint_dir[2] ,checkp_to_eval[2]) #05
    model_expert_4 = load_chkp(model_expert_4, checkpoint_dir[3] ,checkp_to_eval[3]) #06
    model_expert_5 = load_chkp(model_expert_5, checkpoint_dir[4] ,checkp_to_eval[4]) #07
    model_expert_6 = load_chkp(model_expert_6, checkpoint_dir[5] ,checkp_to_eval[5]) #08
    """
    
    expert_models = {
        0: load_chkp(model_expert_1, checkpoint_dir[0] ,expert_checkp[0]), #00,
        1: load_chkp(model_expert_2, checkpoint_dir[1] ,expert_checkp[1]), #02,
        2: load_chkp(model_expert_3, checkpoint_dir[2] ,expert_checkp[2]), #05,
        3: load_chkp(model_expert_4, checkpoint_dir[3] ,expert_checkp[3]), #06,
        4: load_chkp(model_expert_5, checkpoint_dir[4] ,expert_checkp[4]), #07,
        5: load_chkp(model_expert_6, checkpoint_dir[5] ,expert_checkp[5]) #08
    }
    print("Expert models loaded")
    
    #import pdb; pdb.set_trace()
    ###########################
    # load gate

    from train_relu import ExpertClassifier

    # Define the model with the same architecture
    gate_model = ExpertClassifier()
    
    # Load the saved weights
    print("load ","expert_router.pth")
    gate_model.load_state_dict(torch.load("expert_router_22.pth"))
    #gate_model.load_state_dict(torch.load("expert_router_22.pth"))
    gate_model.to(device="cuda")
    gate_model.eval()  # Set the model to evaluation mode
    
    print("Gate model loaded")
    list_seq = [0, 2, 5, 6, 7, 8] 
    
    def predict_expert(model, feature_vector, device):
        with torch.no_grad():
            feature_vector = feature_vector.to(device).unsqueeze(0)  
            output = model(feature_vector)
            predicted_expert_idx = torch.argmax(output).item()
    
            # proba
            m = torch.nn.Softmax(dim=1)
            prob_seq = m(output)
    
        return list_seq[predicted_expert_idx], predicted_expert_idx, output[0][predicted_expert_idx], prob_seq[0][predicted_expert_idx] 

    
    ###########################
    
    def get_seen_ids(num_queries, positions_database):
        seen_ids = []
        for query_idx in range(num_queries):  
            #import pdb; pdb.set_trace()
            query_pose = positions_database[query_idx]
            seen_poses.append(query_pose) # 0 à 1101
    
            if query_idx%5 == 0:
                continue
    
            seen_ids.append('%06d' % query_idx) # 0 à 880
            db_seen_ids = np.copy(seen_ids)
        
            if eval_set.labeltype == 'gps' :
                print(query_idx, eval_set.label2gps(query_idx))
            elif eval_set.labeltype == 'hilbert' :
                print(query_idx, eval_set.label2hilbert(query_idx))
        return seen_poses, db_seen_ids

    db_seen_ids = get_seen_ids(num_queries, positions_database)
    
    seen_poses_1, db_seen_ids_1 = get_seen_ids(4541, positions_database_1)
    seen_poses_2, db_seen_ids_2 = get_seen_ids(4661, positions_database_2)
    seen_poses_3, db_seen_ids_3 = get_seen_ids(2761, positions_database_3)
    seen_poses_4, db_seen_ids_4 = get_seen_ids(1101, positions_database_4)
    seen_poses_5, db_seen_ids_5 = get_seen_ids(1101, positions_database_5)
    seen_poses_6, db_seen_ids_6 = get_seen_ids(4071, positions_database_6)
            
    expert_poses = {
            0: seen_poses_1, #00,
            1: seen_poses_2, #02,
            2: seen_poses_3, #05,
            3: seen_poses_4, #06,
            4: seen_poses_5, #07,
            5: seen_poses_6 #08
        }
    
    ### Restrict decod vocab
    ID_MAX_LENGTH=18

    LIK = []
    
    if eval_set.labeltype == 'gps' :
        for ii in db_seen_ids : LIK.append(tokenizer(eval_set.label2gps(ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids) 
    elif eval_set.labeltype == 'hierarchical' :
        LIK_1, LIK_2, LIK_3, LIK_4, LIK_5, LIK_6 = [], [], [], [], [], []
        
        for ii in db_seen_ids_1 : print(expert_tokenizer[0](expert_hierar_label[0](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
            
        for ii in db_seen_ids_1 : LIK_1.append(expert_tokenizer[0](expert_hierar_label[0](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        for ii in db_seen_ids_2 : LIK_2.append(expert_tokenizer[1](expert_hierar_label[1](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        for ii in db_seen_ids_3 : LIK_3.append(expert_tokenizer[2](expert_hierar_label[2](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        for ii in db_seen_ids_4 : LIK_4.append(expert_tokenizer[3](expert_hierar_label[3](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        for ii in db_seen_ids_5 : LIK_5.append(expert_tokenizer[4](expert_hierar_label[4](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        for ii in db_seen_ids_6 : LIK_6.append(expert_tokenizer[5](expert_hierar_label[5](ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)

        expert_LIK = {
            0: LIK_1, #00,
            1: LIK_2, #02,
            2: LIK_3, #05,
            3: LIK_4, #06,
            4: LIK_5, #07,
            5: LIK_6 #08
        }

        expert_3d_enc = {
            0: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth', #00,
            1: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints//kitti_10cm_loo/2021-09-14_05-55-20_3n24h_Kitti_v10_q29_10s2_262448.pth', #02,
            2: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints//kitti_10cm_loo/2021-09-14_06-11-58_3n24h_Kitti_v10_q29_10s5_262449.pth', #05,
            3: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints//kitti_10cm_loo/2021-09-14_06-43-47_3n24h_Kitti_v10_q29_10s6_262450.pth', #06,
            4: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints//kitti_10cm_loo/2021-09-14_08-34-46_3n24h_Kitti_v10_q29_10s7_262451.pth', #07,
            5: '/lustre/fswork/projects/rech/dki/ujo91el/checkpoint/LoGG3D-NET/checkpoints//kitti_10cm_loo/2021-09-14_20-28-22_3n24h_Kitti_v10_q29_10s8_263169.pth' #08
        }
        
        #for ii in db_seen_ids : LIK.append(tokenizer(eval_set.get_hierarchical_label(ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
            
   
    elif eval_set.labeltype == 'hilbert' :
        for ii in db_seen_ids : LIK.append(tokenizer(eval_set.label2hilbert(ii),padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
    else :
        for ii in db_seen_ids : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
  

    """
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
    """
    
    #def restrict_decode_vocab_vf(batch_idx, prefix_beam):
        #pfb = tuple(prefix_beam.cpu().numpy())  # Convert tensor to tuple
        #return prefix_dict.get(pfb, [102])

    def create_restrict_decode_vocab_vf(prefix_dict):
        def restrict_decode_vocab_vf(batch_idx, prefix_beam):
            pfb = tuple(prefix_beam.cpu().numpy())  # Convert tensor to tuple
            return prefix_dict.get(pfb, [102])  # Use the specific expert dictionary
        return restrict_decode_vocab_vf


    """
    def restrict_decode_vocab_v4(batch_idx, prefix_beam): # to verify dictionnary equivalence with tok_id_ok
        TOK_ID_OK = []
        sz = len(prefix_beam)
        pfb = prefix_beam.cpu().numpy()
        for kk in LIK :
            if kk[:sz] == pfb.tolist()  :
                TOK_ID_OK.append(kk[sz])
        if len(TOK_ID_OK) == 0 :
            TOK_ID_OK.append(102)
        
        #pfb2 = tuple(prefix_beam.cpu().numpy())  # Convert tensor to tuple
        #TOK_ID_OK2 = prefix_dict.get(pfb2, [102])

        pfb2 = tuple(prefix_beam.cpu().numpy())  # Convert tensor to tuple
        TOK_ID_OK3 = prefix_dict2.get(pfb2, [102])

        #print("set(TOK_ID_OK2) - set(TOK_ID_OK) " , set(TOK_ID_OK2) - set(TOK_ID_OK)) 
        #print("set(TOK_ID_OK3) - set(TOK_ID_OK) " , set(TOK_ID_OK3) - set(TOK_ID_OK)) 
        #print("set(TOK_ID_OK) - set(TOK_ID_OK3) " , set(TOK_ID_OK) - set(TOK_ID_OK3)) 
        if  set(TOK_ID_OK3) != set(TOK_ID_OK):
            import pdb; pdb.set_trace()
        return TOK_ID_OK3
    """
    
    len_eval = 0


    ######################################################
    with open("/lustre/fswork/projects/rech/dki/ujo91el/code/these_place_reco/LoGG3D-Net/config/kitti_tuples/is_revisit_D-3_T-30.json") as f:
        data = json.load(f)
   
    list_seq = [0, 2, 5, 6, 7, 8] 
    count = -1
    hit, num = 0, 0
    seen_proba = []

    for seq in list_seq:
        seq_str = f"{seq:02d}"
        count  += 1
        for query_idx in range(len(data[seq_str])):
   
            if query_idx % 10 != 0:
                continue  
            print("query_idx ", query_idx)
    ######################################################


            seen_pose = expert_poses[count]
            
            query_pose = seen_pose[query_idx]
      
            len_eval += 1
            is_revisit = is_revisit_list[query_idx]
            
            # Find top-1 candidate.
            nearest_idx = 0
            min_dist = math.inf
           
            beam_ids = None
            if eval_set.labeltype != 'log3dnet' or True :
                
                prep_timer.tic()   
                input_data = expert_data_collator[count](torch.utils.data.Subset(expert_eval_subset[count],range(query_idx, query_idx+1)))  
                #input_data = data_collator(torch.utils.data.Subset(eval_subset,range(query_idx, query_idx+1)))  
                prep_timer.toc()

                num +=1

                seq_str =  '%02d' % list_seq[count]
                root_path = "/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/"
                desc_path = os.path.join(root_path, seq_str, "logg_desc")
                file_path = os.path.join(desc_path, f"{query_idx:06d}.pt")
                print("file_path ", file_path)
                test_feature = torch.load(file_path).to(torch.float32)
    
                best_expert, expert_seq, score, prob = predict_expert(gate_model, test_feature, 'cuda')
                print(file_path)
                #print(f"Predicted expert: {best_expert}, Expected expert: {int(seq_str)}, Score: {score}, proba: {prob} ")
    
                
    
                def load_json(file_path):
                    """Load JSON file safely and return data."""
                    with open(file_path, "r") as f:
                        return json.load(f)
                
                def compute_distances(query_pose, place_candidates):
                    """Compute distances and return distance lists."""
                    p_dists = np.linalg.norm(query_pose - np.array(place_candidates), axis=1)
                    hits_clos = ['0' if x > 3 else '1' for x in p_dists]
                    return p_dists, hits_clos
                
                def min_distance_beam(query_pose, label_ids, data, seen_poses):
                    """Compute minimum beam distance."""
                    return min(np.linalg.norm(query_pose[:2] - seen_poses[int(data[kk])][:2]) for kk in label_ids)
                
    
                def call_expert(model, input_data, ID_MAX_LENGTH, num_beams, restrict_decode_vocab_vf, encoder_chekp=None):
                    input_data['lidar_values']['encoder_chekp'] = encoder_chekp
                    desc_timer.tic()
                    with torch.no_grad():
                        batch_beams_dict = model.generate(
                                #pixel_values=None,
                                #pixel_values=input_data['pixel_values'],
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
                                prefix_allowed_tokens_fn=restrict_decode_vocab_vf,
                                return_dict_in_generate=True,                
                                output_scores = True,
                                #encoder_chekp = encoder_chekp
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
                    print("label_ids ", label_ids)
    
                    return label_ids, ids, vv, res, seq_score, batch_beams
    
                def get_nearest(label_ids, ids, vv, res, seq_score, batch_beams, expert_seq):
                    # Load JSON files when needed
                    gps_data = load_json(sequence_path + "gps.json") if eval_set.labeltype == 'gps' else None
                    hilbert_data = load_json(sequence_path + "hilbert.json") if eval_set.labeltype == 'hilbert' else None
                    if eval_seq =="06":
                        hilbert_data = load_json(sequence_path + "hilbert_16.json") if eval_set.labeltype == 'hilbert' else None
    
                    # Process based on label type
                    if eval_set.labeltype in ['gps', 'hilbert']:
                        data = gps_data if eval_set.labeltype == 'gps' else hilbert_data  
                        #nearest_ids = [data[label_id] for label_id in label_ids]
                        nearest_ids = [data.get(label_id, -1) for label_id in label_ids]
                        nearest_idx = nearest_ids[0]
                    
                    elif eval_set.labeltype == 'hierarchical':
                        #nearest_idx = int(eval_set.inv_hierarchical_label[label_ids[0]])
                        #nearest_ids = [int(eval_set.inv_hierarchical_label[label_id]) for label_id in label_ids]
                        sequence_path = os.path.join(root_path, f"{expert_seq:02d}")
                        hierar_data = load_json(sequence_path + "/hierarchical.json")
                        nearest_ids = [int(hierar_data.get(label_id, -1)) for label_id in label_ids]
                        print("nearest_ids  ", nearest_ids )
                        #nearest_ids_og = [int(eval_set.inv_hierarchical_label.get(label_id, -1)) for label_id in label_ids]
                        #print("nearest_ids_og ", nearest_ids_og)
                        nearest_idx = nearest_ids[0]
                        #import pdb; pdb.set_trace()
                    else:
                        nearest_ids = label_ids
                        nearest_idx = nearest_ids[0]
                    return nearest_ids, nearest_idx
    
                def get_dists(seen_poses, nearest_ids):
                    # Common processing logic
                    #place_candidate = seen_poses[int(nearest_idx)]
                    place_candidates = [seen_poses[int(nearest_id)] for nearest_id in nearest_ids]
                    place_candidate = place_candidates[0]
                    #place_candidate array([270.7098619,  27.72478  ,  21.6189174])
                    # Compute distances
                    p_dist = np.linalg.norm(query_pose - place_candidate)
                    p_dists, hits_clos = compute_distances(query_pose, place_candidates)
                    print("p_dist, p_dists, hits_clos ",  p_dist, p_dists, hits_clos)
                    #import pdb; pdb.set_trace()
                    return p_dist, p_dists, hits_clos
    
                def DSI_3D(model, input_data, ID_MAX_LENGTH, num_beams,  prefix_dict, restrict_decode_vocab_vf, seen_poses, expert_seq, encoder_chekp=None):
                    label_ids, ids, vv, res, seq_score, batch_beams = call_expert(model, input_data, ID_MAX_LENGTH, num_beams, restrict_decode_vocab_vf, encoder_chekp)
                    nearest_ids, nearest_idx = get_nearest(label_ids, ids, vv, res, seq_score, batch_beams, expert_seq)
                    p_dist, p_dists, hits_clos = get_dists(seen_poses, nearest_ids)
                    return label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos
    
    
                ##########################################################
                """
                # call all model
                prefix_dict = expert_prefix[0]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[0], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_1,0)
    
                prefix_dict = expert_prefix[1]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[1], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_2, 2)
                
                
                prefix_dict = expert_prefix[2]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[2], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_3, 5)
                
                prefix_dict = expert_prefix[3]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[3], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_4, 6)
                
                prefix_dict = expert_prefix[4]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[4], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_5, 7)
                
                prefix_dict = expert_prefix[5]
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(expert_models[5], input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_poses_6, 8)
                """
                ##########################################################
    
                #########################################################
                
                # best model
                model = expert_models[expert_seq]
                tokenizer = expert_tokenizer[expert_seq] 
                LIK = expert_LIK[expert_seq]
                prefix_dict = expert_prefix[expert_seq]
                encoder_chekp = expert_3d_enc[expert_seq]

                # gt
                seen_pose = expert_poses[expert_seq]
                
                
                restrict_decode_vocab_vf = create_restrict_decode_vocab_vf(prefix_dict)
                label_ids, ids, vv, res, seq_score, batch_beams, nearest_ids, nearest_idx, p_dist, p_dists, hits_clos = DSI_3D(model, input_data, ID_MAX_LENGTH, num_beams, prefix_dict, restrict_decode_vocab_vf, seen_pose, best_expert, encoder_chekp)

                ids, vv, res, batch_beams = None, None, None, None
                #import pdb; pdb.set_trace()
                seen_proba.append(prob.cpu().numpy())
                
                print(f"Predicted expert: {best_expert}, Expected expert: {int(seq_str)}, Score: {score}, proba: {prob} ")


                if best_expert ==  int(seq_str):
                    hit += 1
                else:
                    p_dist = 1000
                    hits_clos = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
                    print("p_dist, hits_clos ",  p_dist, hits_clos)
                    #import pdb; pdb.set_trace()
                    
    
                ##########################################################
                
                #print("label_ids ", label_ids)
                #import pdb; pdb.set_trace()
                p_dist_mean = 0
                p_dist_beam = 0
                beam_ids = None
                    
    
                #########################################################
                #hierar00 =load_json("/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/00/hierarchical.json")
                #hierar02 =load_json("/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/02/hierarchical.json")
                #hierar22 =load_json("/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/datasets/sequences/22/hierarchical.json")
                #########################################################
                
                
                # Compute beam distance where applicable
                """
                if eval_set.labeltype in ['gps', 'hilbert']:
                    p_dist_beam = min_distance_beam(query_pose, label_ids, data, seen_poses)
                elif eval_set.labeltype == 'hierarchical':
                    #p_dist_beam = min(np.linalg.norm(query_pose - seen_poses[int(eval_set.inv_hierarchical_label[kk])]) for kk in label_ids)
                    p_dist_beam = None 
                else:
                    p_dist_beam = None  # No beam distance for the default case
                """
                    
                beam_ids = None
                min_dist = -seq_score[0][0].cpu().numpy()
    
                
            else :
                feat_dists = cdist(global_descriptor, db_seen_descriptors,
                                   metric='cosine').reshape(-1)
                min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
                # ret_timer.toc()
                place_candidate = seen_poses[nearest_idx]
                p_dist = np.linalg.norm(query_pose - place_candidate)
      
            #np.linalg.norm(seen_poses[120] - seen_poses[867])
            
            is_revisit = is_revisit_list[query_idx]
            is_correct_loc = 0
            is_correct_loc_beam = np.zeros(num_beams)
            
            ######################################################################################
            # Hitsscores only revisited (OG)
            ######################################################################################
        
            if is_revisit:
                #import pdb; pdb.set_trace()        
                num_revisits += 1
                if p_dist <= revisit_criteria:
                    num_correct_loc += 1
                    is_correct_loc = 1
          
            ######################################################################################
            # Hitsscores all
            ######################################################################################
            if p_dist <= revisit_criteria:
                num_correct_loc_all += 1
                
            if '1' in hits_clos:
                hit_at_10 += 1 
            ######################################################################################
    
            def log_query_info(labeltype, query_idx, nearest_idx, nearest_ids, label_ids, is_revisit, is_correct_loc, p_dist, min_dist, eval_set):
                """Logs query information based on labeltype."""
                labeltype_mapping = {
                    'hierarchical': ('id', eval_set.hierarchical_label.get(str(query_idx), 'N/A')),
                    #'gps': ('id', eval_set.label2gps(str(query_idx))),
                    #'hilbert': ('id', eval_set.label2hilbert(str(query_idx)))
                }
            
                if labeltype in labeltype_mapping:
                    label_name, query_label = labeltype_mapping[labeltype]
                    logging.info(
                        f"{label_name}:{query_idx} {label_name}_val:{query_label} Top1_{labeltype}:{label_ids[0]} "
                        f"Top1_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} "
                        f"p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} "
                    )
                    logging.info(
                        f"{label_name}:{query_idx} {label_name}_val:{query_label} TopN_{labeltype}:{label_ids} TopN_id:{nearest_ids} "
                    )
                else:
                    logging.info(
                        f"id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} "
                        f"p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} "
                    )
            
            # Call the function
            log_query_info(eval_set.labeltype, query_idx, nearest_idx, nearest_ids, label_ids, is_revisit, is_correct_loc, p_dist, min_dist, eval_set)
    
    
            #saved predictions
            dictio.append({"query_idx":query_idx,"Top1_id":nearest_idx, "is_rev":is_revisit, "loc_ok_1":is_correct_loc, "p_dist":p_dist, "min_dist":min_dist})
        
            if beam_ids is not None : 
                print('[beam:' + ' '.join(map(str, is_correct_loc_beam.astype(int))) + ']')
                #print('[beam:' + ' '.join(f'{x:.2f}' for x in p_dist_beam) + ']')
                     
        
            if min_dist < min_min_dist:
                min_min_dist = min_dist
            if min_dist > max_min_dist:
                max_min_dist = min_dist


        def Evaluate_top_1_candidate(num_thresholds, thresholds, min_dist, p_dist, revisit_criteria, not_revisit_criteria, is_revisit, num_true_positive, num_false_positive, num_true_negative, num_false_negative):
            for thres_idx in range(num_thresholds):
                threshold = thresholds[thres_idx]
                
                if(min_dist < threshold):  # Positive Prediction
                    if p_dist <= revisit_criteria :
                        num_true_positive[thres_idx] += 1
                    elif p_dist > not_revisit_criteria:
                        num_false_positive[thres_idx] += 1
      
                        
                else:  # Negative Prediction
                    if p_dist > revisit_criteria :
                        num_true_negative[thres_idx] += 1
                    elif p_dist <= not_revisit_criteria:
                        num_false_negative[thres_idx] += 1
            
            
            return num_true_positive, num_false_positive, num_true_negative, num_false_negative 
    
    
        ######################################################################################
        # classification binaire avec zones grises
        # not_revisit_criteria = 20
        #num_true_positive, num_false_positive, num_true_negative, num_false_negative = Evaluate_top_1_candidate(num_thresholds, thresholds, min_dist, p_dist, revisit_criteria, not_revisit_criteria, is_revisit, num_true_positive, num_false_positive, num_true_negative, num_false_negative)
        ######################################################################################
        # classification binaire sans zones grises
        # not_revisit_criteria = revisit_criteria = 3
        #num_true_positive_3m, num_false_positive_3m, num_true_negative_3m, num_false_negative_3m = Evaluate_top_1_candidate(num_thresholds, thresholds, min_dist, p_dist, revisit_criteria, revisit_criteria, is_revisit, num_true_positive_3m, num_false_positive_3m, num_true_negative_3m, num_false_negative_3m)
    
    
        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]
            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= revisit_criteria :
                    num_true_positive[thres_idx] += 1
                    #name_true_positive[thres_idx].append({query_idx:nearest_idx}) 
    
                elif p_dist > not_revisit_criteria:
                    num_false_positive[thres_idx] += 1
                    #name_false_positive[thres_idx].append([f'id:{query_idx} n_id:{nearest_idx} is_rev:{is_revisit} -- loc_ok_1:{is_correct_loc} p_dist:{p_dist:6.2f} min_dist:{min_dist:6.2f} '])  
                    
            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                    #name_true_negative[thres_idx].append({query_idx:nearest_idx}) 
                else:
                    num_false_negative[thres_idx] += 1
                    #name_false_negative[thres_idx].append({query_idx:nearest_idx}) 
    
    
        ######################################################################################
        # classification binaire sans zones grises
        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]  
            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= revisit_criteria :
                    num_true_positive_3m[thres_idx] += 1
                elif p_dist > revisit_criteria:
                    num_false_positive_3m[thres_idx] += 1   
                    
            else:  # Negative Prediction
                if p_dist > revisit_criteria:
                    num_true_negative_3m[thres_idx] += 1    
                elif p_dist <= revisit_criteria:
                    num_false_negative_3m[thres_idx] += 1

    

    ######################################################################################
    ######################################################################################

    def evaluate_classification(num_true_negative, num_false_positive, num_true_positive, num_false_negative, num_thresholds):
        """Evaluates classification metrics and returns F1 max and related statistics."""
        F1max = 0.0
        best_metrics = {}
        Precisions, Recalls = [], []
    
        for ithThres in range(num_thresholds):
            nTN, nFP, nTP, nFN = (
                num_true_negative[ithThres], num_false_positive[ithThres],
                num_true_positive[ithThres], num_false_negative[ithThres]
            )
    
            Precision = nTP / (nTP + nFP) if (nTP + nFP) > 0 else 0.0
            Recall = nTP / (nTP + nFN) if (nTP + nFN) > 0 else 0.0
            F1 = (2 * Precision * Recall / (Precision + Recall)) if (Precision + Recall) > 0 else 0.0
    
            if F1 > F1max:
                F1max = F1
                best_metrics = {"F1_TN": nTN, "F1_FP": nFP, "F1_TP": nTP, "F1_FN": nFN, "F1_thresh_id": ithThres}
    
            Precisions.append(Precision)
            Recalls.append(Recall)
    
        return F1max, best_metrics, Precisions, Recalls
    
    
    def log_evaluation_results(label, num_revisits, num_correct_loc, len_eval, num_correct_loc_all, hit_at_10, min_min_dist, max_min_dist, F1max, best_metrics, prep_timer, desc_timer, ret_timer):
        """Logs evaluation results in a structured format."""
        logging.info(f'{label}')
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(f'percentage_correct_loc: {num_correct_loc * 100.0 / num_revisits:.2f}%')
    
        logging.info(f'num_eval_set: {len_eval}')
        logging.info(f'num_correct_loc_all: {num_correct_loc_all}')
        logging.info(f'percentage_correct_loc_all: {num_correct_loc_all * 100.0 / len_eval:.2f}%')
    
        logging.info(f'hit_at_10: {hit_at_10}')
        logging.info(f'percentage_correct_loc: {hit_at_10 * 100.0 / len_eval:.2f}%')
    
        logging.info(f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(f'F1_TN: {best_metrics["F1_TN"]} F1_FP: {best_metrics["F1_FP"]} F1_TP: {best_metrics["F1_TP"]} F1_FN: {best_metrics["F1_FN"]}')
        logging.info(f'F1_thresh_id: {best_metrics["F1_thresh_id"]}')
        logging.info(f'F1max: {F1max:.4f}')
    
        logging.info('Average times per scan:')
        logging.info(f"--- Prep: {prep_timer.avg:.4f}s Desc: {desc_timer.avg:.4f}s Ret: {ret_timer.avg:.4f}s ---")
        logging.info(f'Average total time per scan: --- {prep_timer.avg + desc_timer.avg + ret_timer.avg:.4f}s ---')
        return
    
    if not save_descriptors:
        # Evaluate standard classification
        F1max, best_metrics, Precisions, Recalls = evaluate_classification(
            num_true_negative, num_false_positive, num_true_positive, num_false_negative, num_thresholds)
    
        log_evaluation_results(
            "Standard Classification", num_revisits, num_correct_loc, len_eval, 
            num_correct_loc_all, hit_at_10, min_min_dist, max_min_dist, 
            F1max, best_metrics, prep_timer, desc_timer, ret_timer)
    
        # Evaluate stricter classification
        F1max, best_metrics, Precisions, Recalls = evaluate_classification(
            num_true_negative_3m, num_false_positive_3m, num_true_positive_3m, num_false_negative_3m, num_thresholds)
    
        log_evaluation_results(
            "More Strict Classification", num_revisits, num_correct_loc, len_eval, 
            num_correct_loc_all, hit_at_10, min_min_dist, max_min_dist, 
            F1max, best_metrics, prep_timer, desc_timer, ret_timer)
    




        F1_thresh_id =best_metrics["F1_thresh_id"]
        
        
        if eval_set.labeltype == 'log3dnet' :
            checkpoint_dir = "/lustre/fswork/projects/rech/xhk/ufm44cu/checkpoints/logg3dnet"
        
        if plot_pr_curve and False:
            plt.figure()
            plt.title('Seq: ' + str(eval_seq) +
                      '    F1Max: ' + "%.4f" % (F1max))
            plt.plot(Recalls, Precisions, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1, 0, 1.1])
            plt.xticks(np.arange(0, 1.01, step=0.1))
            plt.grid(True)
            save_dir = os.path.join(checkpoint_dir_1, 'pr_curves')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_seq = str(eval_seq).split('/')[-1]
            plt.savefig(save_dir + '/' + eval_seq + '.png')

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

    


    ################################################################################
    # re compute binary classification for saving file
    ################################################################################
    
    # Build stat    
    list_true_positive = []
    list_false_positive = []
    list_true_negative = []
    list_false_negative = []
    
    threshold = F1_thresh_id * 0.001 + 0.001
    
    for query_idx in range(len(dictio)):
        if(dictio[query_idx]["min_dist"] < threshold):  # Positive Prediction
            if dictio[query_idx]["p_dist"] <= 3 :
                list_true_positive.append(dictio[query_idx])
            elif dictio[query_idx]["p_dist"] > 3:
                list_false_positive.append(dictio[query_idx])  
                
        else:  # Negative Prediction
            if dictio[query_idx]["p_dist"] > 3 :
                list_true_negative.append(dictio[query_idx])    
            elif dictio[query_idx]["p_dist"] <= 3 :
                list_false_negative.append(dictio[query_idx])

    print(len(list_true_negative),len(list_false_positive), len(list_true_positive), len(list_false_negative))
    
    print("correct prediction (%): ", hit / num)
    print("average proba: ", np.mean(seen_proba) )


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
            if len(dict_to) != 0:
                print("mean ", round(sum(d[str(scor_key)] for d in dict_to) / len(dict_to),round_num))
                print("min ", round(min(dict_to, key=lambda x:x[str(scor_key)])[str(scor_key)],round_num))
                print("max ", round(max(dict_to, key=lambda x:x[str(scor_key)])[str(scor_key)],round_num))
                listr = []
                for k in range(len(dict_to)): listr.append(dict_to[k][str(scor_key)])
                print("std ", round(np.std(listr),round_num))
            count += 1


    def compute_statistics(dict_to, key, round_num):
        """
        Compute and return statistics (mean, min, max, std) for a given key in the dictionary list.
        """
        values = [d[key] for d in dict_to]
        stats = {
            "mean": round(np.mean(values), round_num),
            "min": round(min(values), round_num),
            "max": round(max(values), round_num),
            "std": round(np.std(values), round_num),
        }
        return stats

    # Class and corresponding lists
    print_class = ["TP", "FP", "FN", "TN"]
    all_lists = [list_true_positive, list_false_positive, list_false_negative, list_true_negative]
    
    # Iterate over each scoring key and compute statistics
    for scor_key in ['p_dist', 'min_dist']:
        print(scor_key)
        round_num = 2 if scor_key == 'p_dist' else 4
    
        for class_name, dict_list in zip(print_class, all_lists):
            print(class_name)
            if len(dict_list) > 0:
                stats = compute_statistics(dict_list, scor_key, round_num)
                for stat_name, value in stats.items():
                    print(f"{stat_name}: {value}")
            else:
                print("No data available")

    
    import pdb; pdb.set_trace() 

    
    
    return F1max                    
