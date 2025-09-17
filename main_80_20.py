## GD-MAE
import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from extern.log3dnet.SOP import SOP
from collections import Counter
from dataclasses import replace 
import time
# 
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

import hostlist

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter 
import copy 
import traceback
import logging

from extern.pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from extern.pcdet.datasets import build_dataloader
from extern.pcdet.utils import common_utils
#from extern.train_utils.optimization import build_optimizer, build_scheduler
#from extern.train_utils.train_utils import train_model
import numpy as np

## Blip2
import requests
from PIL import Image
from transformers import AutoProcessor,AutoModel, AutoConfig, AutoTokenizer, TrainingArguments , HfArgumentParser
from extern.blip2.modeling_blip_2 import Blip2ModelQuerryLearning
from extern.blip2.processing_blip_2 import Blip2Processor
from transformers import BertTokenizer, BertModel,BertLMHeadModel,MT5Tokenizer
from extern.blip2.modeling_bert_generation   import BertGenerationDecoder
from transformers import  GPTQConfig

## DSI QG
from dataclasses import dataclass
from transformers.trainer import Trainer
from transformers import PreTrainedTokenizer, DataCollatorWithPadding,PretrainedConfig
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import   MT5ForConditionalGeneration
from extern.git.modeling_git import GitModel,GitForCausalLM

from evaluate_log3dnet_80_20 import eval_log3dnet
#from evaluate_overfit import eval_overfit
#from compute_hierarchical_index import compute_hierarchical_clustering

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import TrainerCallback

##################################
# read pos
##################################
from module_loader_kitti_pose import * # add for more metrics
import math
import gc

WORK_PATH = os.getenv('WORKSF')

## ====  Usefull stuff =======
def forward_nan_hook(self, inp, output):
    print("not implemented yet")

def backward_nan_hook(name):
    def hook(module, grad_input, grad_output):
        if (len(grad_input) == 0 or len(grad_output) == 0) :
            return 
        if grad_input[0] == None or grad_output[0] == None :
            return 
        if (torch.isnan(grad_input[0]).any() or
            torch.isnan(grad_output[0]).any()) :

            print("\n")
            raise RuntimeError(f"Found NAN in gradient")
    return hook


def get_pts_for_plot(query_idx,eval_seq,tfs,pose) :

    kitti_dir = WORK_PATH+"/datas/datasets/"
    fname = kitti_dir + 'sequences/'+eval_seq+'/velodyne/'+'%06d' % query_idx + '.bin'
    #load points
    xyz = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    # every possible positions
    x, z, y = pose[:,0], pose[:,1], pose[:,2]

    # rotation 1
    out = np.zeros((len(xyz), 3))
    mat = tfs[query_idx][:3,:3]
    for i in range (len(xyz)):
        out[i] = ( mat @ xyz[i][:3] ) 
    xyzr = out

    # translation
    pose_q = pose[query_idx]
    pose_q[[1,2]] =  pose_q[[2,1]]
    xyzrf = xyzr[:, :3] + pose_q

    return xyzrf
    
def print_loader(loader,lab) :
    do_dump_image = False
    lit = iter(loader)

    eval_seq = cfg['DATA_CONFIG']['SEQ']
    kitti_dir = WORK_PATH+"/datas/datasets/"
    print("")
    print("=========  loader " + lab + " ===========")
    print("ln : " + str(len(loader)))
    acc = 0
    os.makedirs("plot_" + lab, exist_ok=True)
    sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
    tfs, pose = load_poses_from_txt(sequence_path + 'poses.txt')
    for ii in lit :
        print("id:" + str(ii['id']) + " gt:" + str(ii['gt']) + " labels:" + str(ii['labels']) ) #+ " gps_label:" + str(ii['gps']))
        acc = acc+1

        if do_dump_image : 
            xyzrf = get_pts_for_plot(int(ii['id']),eval_seq,tfs,pose)
            if int(ii['gt']) > 0:
                xyzrf_gt = get_pts_for_plot(int(ii['gt']),eval_seq,tfs,pose)
            x, z, y = pose[:,0], pose[:,1], pose[:,2]
            plt.figure()
            plt.scatter(xyzrf[:, 0], xyzrf[:, 2], c='b', s=0.05,marker='x')
            if int(ii['gt']) > 0:
                plt.scatter(xyzrf_gt[:, 0], xyzrf_gt[:, 2], c='r', s=0.05,marker='o')
            plt.scatter(x,z,c='g', s=0.1)
            plt.xlabel("X")
            plt.ylabel("z")
            plt.title("query "+str(ii['id']) )
            plt.axis('equal')
            plt.savefig('plot_' + lab + '/query_'+str(ii['id']) +'.png',dpi=600)
        
        #import pdb; pdb.set_trace()
        if(acc > 64) :
            print("....")
            break
    print("========= end loader ===========")


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'image_shape', 'image_pad_shape', 'image_rescale_shape','labels','index','input_ids','transformation_3d_list','transformation_3d_params','transformation_2d_list','transformation_2d_params','batch_size','gt','gts','id','gps','hilbert', 'id_pcd_positif','id_pcd_negatif','other_id_pcd_negatif',  'lidar_values', 'lidar_values_load']: #ajout key truth
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        
        if not str(m).startswith("Linear8bitLt") :
            m.reset_parameters()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--model_name', type=str, default="git", help='checkpoint to start from')
    parser.add_argument('--use_sop', type=str, default="True", help='')    
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training GD-MAE')
    parser.add_argument('--eval_steps', type=int, default=100, required=False, help='batch size for training DSI')
    parser.add_argument('--evaluation_strategy', type=str, default="steps", required=False, help='batch size for training DSI')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False, help='batch size for training DSI')
    parser.add_argument('--warmup_steps', type=int, default=1000, required=False, help='batch size for training DSI')
    parser.add_argument('--save_steps', type=int, default=0, required=False, help='batch size for training DSI')
    parser.add_argument('--logging_steps', type=int, default=1, required=False, help='batch size for training DSI')
    parser.add_argument('--train_batch_size', type=int, default=32, required=False, help='batch size for training DSI')
    parser.add_argument('--git_checkpoint', type=str, default=None, help='specify the config for training')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='specify the config for training')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='specify the config for training')

    
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, required=False, help='batch size for training DSI')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, required=False, help='batch size for training DSI')


    parser.add_argument('--do_train', type=str, default="False", help='')
    parser.add_argument('--do_eval', type=str, default="False", help='')
    parser.add_argument('--do_eval_partial', type=str, default="False", help='')
    parser.add_argument('--do_preprocess', type=str, default="False", help='')
    parser.add_argument('--do_dump_dict_gt', type=str, default="False", help='')
    
    parser.add_argument('--adam_epsilon', type=float, default=1e-05, required=False, help='adam_epsilon')
    parser.add_argument('--dataset_train_len', type=int, default=64, required=False, help='adam_epsilon')
    parser.add_argument('--dataset_eval_len', type=int, default=16, required=False, help='adam_epsilon')
    parser.add_argument('--learning_rate', type=float, default=1e-07, required=False, help='adam_epsilon')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--dispatch_batches', type=bool, default=True, required=False, help='')
    
    parser.add_argument('--reset_model', type=bool, default=False, required=False, help='')
    parser.add_argument('--weighted_crossentropy', type=bool, default=False, required=False, help='')   
    parser.add_argument('--adam_beta1', type=float, default=0.9, required=False, help='adam_epsilon')
    parser.add_argument('--adam_beta2', type=float, default=0.999, required=False, help='adam_epsilon')
    parser.add_argument('--num_train_epochs', type=int, default=3, required=False, help='adam_epsilon')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', type=int, default=-1, help='seed')    
    #parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=1, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=10, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--remove_unused_columns', type=bool, default=False, help='')

    
    parser.add_argument('--dataloader_pin_memory', type=bool, default=False, help='')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='')
    parser.add_argument('--output_dir', type=str, default=None, help='output_dir')
    parser.add_argument('--log3dnet_dir', type=str, default=None, help='log3dnet')

    # added
    parser.add_argument('--save_hit_file',  type=str, default='hit.txt', help='file with hit score')
    parser.add_argument('--id_max_length', type=int, default=10, required=False, help='adam_epsilon')
    parser.add_argument('--eval_chkt', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", required=False, help='LR scheduler type')
    parser.add_argument('--num_cycles', type=int, default=1, required=False, help='restart cycles')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.sync_bn = args.sync_bn or cfg.OPTIMIZATION.get('SYNC_BN', False)
    

    return args, cfg


##############################################################################
# Optimisé make_compute_metrics
##############################################################################
def make_compute_metrics(tokenizer, logger, rank, positions_database, label_mapping, save_file_name):
    def compute_metrics(eval_preds):
        hit_at_1, hit_at_10 = 0, 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams, skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            query_id = label_mapping.get(label_id, label_id) # (keyname, value=value to return if the specified key does not exist)
            answers_ids = [label_mapping.get(x, x) for x in rank_list]
            
            # Position-based metrics
            label_id_gps = positions_database[int(query_id)]
            rank_list_gps = [positions_database[int(x)] for x in answers_ids]
            rank_list_dist = [
                math.dist(label_id_gps[:2], rank_list_gps[i][:2]) for i in range(len(rank_list_gps))
            ]
            rank_list_dist_filter = [1 if dist <= 3 else 0 for dist in rank_list_dist]

            hits_clos = np.where(np.array(rank_list_dist_filter)[:10] == 1)[0]
            if hits_clos.size > 0:
                hit_at_10 += 1
                if hits_clos[0] == 0:
                    hit_at_1 += 1


        total_predictions = len(eval_preds.predictions)

        #######################################################################
        # save validation metrics
        with open(save_file_name, 'a') as f:
            f.write(str(hit_at_1 / total_predictions ) + " " + str(hit_at_10 / total_predictions ) + "\n")
        #######################################################################
        
        return {
            "Hits@1": hit_at_1 / total_predictions,
            "Hits@10": hit_at_10 / total_predictions,
        }
    
    return compute_metrics

##############################################################################
# Optimisé DSITrainer
##############################################################################
class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, LIK, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        print(" id_max_length ",  id_max_length)
      
        self.id_max_length = id_max_length
        self.LIK = LIK
        self.per_device_train_batch_size = kwds['args'].per_device_train_batch_size
        self.per_device_eval_batch_size = kwds['args'].per_device_eval_batch_size

    def compute_loss(self, model, inputs, return_outputs=False): # 1
        del inputs['ids']
        outputs = model(**inputs)
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss 

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        model.eval()
        
        vv = self.tokenizer.batch_decode(inputs["labels"],skip_special_tokens=True)
        self.ll1 = []

        with torch.no_grad():
            # Beam search parameters
            batch_size = inputs['pixel_values'].size(0)
            nb_beam = self.id_max_length
            inputs['lidar_values']['batch_size'] = self.per_device_eval_batch_size
            
            # Remove ids from inputs
            ids = inputs.pop('ids')
            
            batch_beams_dict = model.generate(
                pixel_values=inputs['pixel_values'],
                lidar_values=inputs['lidar_values'],
                points=None,
                max_length= self.id_max_length, #8
                num_beams=nb_beam, #8
                num_return_sequences=nb_beam, #8
                eos_token_id=self.tokenizer.eos_token_id, #102
                pad_token_id=self.tokenizer.pad_token_id, #0
                bos_token_id=self.tokenizer.bos_token_id, #101
                renormalize_logits=True,
                early_stopping=False, #True,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                return_dict_in_generate=True,                
                output_scores = True,
            )

            # Extract generated sequences and scores
            batch_beams = batch_beams_dict['sequences']
            seq_score = batch_beams_dict['sequences_scores'].reshape(batch_size, nb_beam)

            # Pad sequences to the maximum length
            batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length, self.tokenizer)
            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length, self.tokenizer)
            
            # Reshape beams for batch-wise operations
            batch_beams = batch_beams.reshape(batch_size, nb_beam, -1)
             
            # Optional: Debugging/logging for predictions
            for ii in range(batch_size):
                decoded_labels = self.tokenizer.batch_decode(batch_beams[ii].cpu(), skip_special_tokens=True)
                print(f"IDs: {ids[ii]}")
                print(f"Labels: {self.tokenizer.decode(inputs['labels'][ii], skip_special_tokens=True)}")
                print(f"Beams: {decoded_labels}")
                print(f"Scores: {seq_score[ii]}")
                print("----")

        return None, batch_beams, inputs['labels'] # loss, logits, labels


    def _pad_tensors_to_max_len(self, tensor, max_length, tokenizer):
        """
        Pads tensor to a specified maximum length using the pad token ID.
        """
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        tensor[tensor == -100] = pad_token_id  # Replace masked tokens
        padded_tensor = pad_token_id * torch.ones(
            (tensor.size(0), max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, :tensor.size(1)] = tensor
        return padded_tensor

@dataclass

##############################################################################
# Optimisé IndexingCollator
##############################################################################
class IndexingCollator(DataCollatorWithPadding):
    def __init__(self, label_tokenizer, padding, id_max_length, processor, batch_size):
        super().__init__(label_tokenizer, padding)
        self.processor = processor
        self.batch_size = batch_size
        self.id_max_length = id_max_length
        self.tokenizer = label_tokenizer
        
    def __call__(self, features):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract features
        input_ids = torch.vstack([x['input_ids'] for x in features])
        labels = input_ids.clone()
        ids = [x['id'] for x in features]
        attention_mask = torch.vstack([x['attention_mask'] for x in features])
        pixel_values = torch.cat([x['pixel_values'] for x in features], dim=0).to(device=device)

        # Process `attention_mask` and `input_ids`
        attention_mask[input_ids == self.tokenizer.eos_token_id] = 0
        input_ids[input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id

        # Prepare `inputs` dictionary
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            'ids': ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask.to(device=device),
        }

        # Process LIDAR values
        lidar_values = self._prepare_lidar_values(features, device)
        inputs['lidar_values'] = lidar_values
        
        # Load LIDAR data to GPU if available
        if device == "cuda":
            load_data_to_gpu(inputs['lidar_values'])
            
        return inputs

    def _prepare_lidar_values(self, features, device):
        lidar_val = {'batch_size': self.batch_size}
        
        feature_dict = {k: [x[k] for x in features] for k in features[0].keys()}
        
        for key, val in feature_dict.items():
            if key == 'points':
        
                padded_points = [torch.nn.functional.pad(
                    torch.tensor(coor, dtype=torch.float32),
                    (0, 0, 1, 0), 
                    value=i  
                ) for i, coor in enumerate(val)]
                lidar_val[key] = torch.cat(padded_points, dim=0).to(device)
            elif key == 'desc':
                lidar_val[key] = val 
            else:
                lidar_val[key] = np.stack(val, axis=0)  
        
        return lidar_val
        ##############################################################################
        ##############################################################################
        
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_name="", threshold=1):
        # metric_name can be any key returned in the evaluation logs (e.g., 'eval_loss', 'eval_accuracy')
        self.metric_name = metric_name
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Check if the evaluation metric is higher than the threshold
        eval_metric = metrics.get(self.metric_name)
        if eval_metric and eval_metric > self.threshold:
            print(f"Stopping training early! {self.metric_name} = {eval_metric}")
            control.should_training_stop = True



def Print_active_layers_git(args, logger, model_dsi, model_name):
    grad_module_name = []
    if model_name == "git" :
        for name, param in model_dsi.named_parameters():
            if (
                name.startswith("vision_model") or
                name.startswith("language_model") or
                name.startswith("git.image_encoder") or
                name.startswith("git.visual_projection") or
                name.startswith("git.embeddings.word_embeddings.weight") or
                name.startswith("git.embeddings.position_embeddings.weight")                    
            ) :
                param.requires_grad = False
            if (
                name.startswith("VOID") 
            ) :
                param.requires_grad = True
            if param.requires_grad == True  :
                grad_module_name.append(name)
                if args.local_rank == 0 :
                    logger.info(name + "\t =>" + str(param.requires_grad))
                    
       
    # Print active layers blip2               
    if model_name == "blip2" :
        for name, param in model_dsi.named_parameters():
            if (
                #name.startswith("bert_model") or
                name.startswith("vision_model") or
                name.startswith("language_model") or
                #name.startswith("qformer.input_embeddings") or
                name.startswith("qformer.input_embeddings.word_embeddings.weight") or 
                name.startswith("qformer.input_embeddings.position_embeddings.weight")                     
            ) :
                param.requires_grad = False
            if (
                name.startswith("qformer.input_embeddings.LayerNorm") or
                name.startswith("qformer.input_embeddings.dropout") or                 
                name.startswith("itm_head") or
                #name.startswith("bert_model.lm_head.bias")  or
                name.startswith("qformer.output_embeddings") or
                name.startswith("text_projection") or
                name.startswith("vision_projection") or
                name.startswith("language_model.lm_head")  or
                name.startswith("query_tokens") 
            ) :
                param.requires_grad = True

            if (args.git_checkpoint is not None) :
                if (
                name.startswith("lidar_model") or
                name.startswith("qformer.output_embeddings")or
                name.startswith("qformer.input_embeddings")
                ) :
                    param.requires_grad = False
            if param.requires_grad == True  :                
                grad_module_name.append(name)
                if args.local_rank == 0 :
                     logger.info(name + "\t =>" + str(param.requires_grad))
    return



def main():
    ### ===== START ===========
    # parametres
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args, cfg = parse_config()
    ID_MAX_LENGTH = args.id_max_length 
    MAX_LENGTH = ID_MAX_LENGTH

    model_name = args.model_name
    dataset_train_len = args.dataset_train_len
    
    dataset_eval_len = args.dataset_eval_len

    checkp_to_eval = args.eval_chkt
    
    do_overfit = True
    random_seed = int(args.fix_random_seed)
    do_use_sop = eval(args.use_sop)
    if args.launcher == "pytorch" : 
        args.local_rank = int(os.environ['LOCAL_RANK'])


    sequence_path = cfg['DATA_CONFIG']['DATA_PATH'] + cfg['DATA_CONFIG']['SEQ']
    
    ##########################################################################################
    save_file_name = args.save_hit_file # new for naming file hit score
    with open(save_file_name, 'w') as f:
        print('create and reset the txt output file ', save_file_name)
    f.close()
    ##########################################################################################

    do_train = eval(args.do_train)
    do_eval = eval(args.do_eval)
    do_eval_partial = eval(args.do_eval_partial)
    do_preprocess = eval(args.do_preprocess)
    do_dump_dict_gt = eval(args.do_dump_dict_gt)

    print("============================================")
    print("do_eval:"  + str(do_eval))
    print("do_train:"  + str(do_eval))
    print("do_eval_partial:"  + str(do_eval_partial))
    print("do_preprocess: "  + str(do_preprocess))
    print("do_dump_dict_gt: "  + str(do_dump_dict_gt))
    print("============================================")

    ### ==== ARGUMENT PARSER  =====
    ## T5 Args parser 
    parser = HfArgumentParser((TrainingArguments,))
  
    ## GD-MAE parser
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if random_seed > 0 :
        common_utils.set_random_seed(random_seed)
    print("RANDOM SEED:" + str(random_seed))
    training_args, remaiening = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    print("training_args.lr_scheduler_type ", training_args.lr_scheduler_type)
    print("training_args.warmup_ratio:", training_args.warmup_ratio)

    batch_size = training_args.per_device_train_batch_size
    ori_train_batch_size  = training_args.per_device_train_batch_size
    ori_eval_batch_size  = training_args.per_device_eval_batch_size
    args.batch_size = batch_size

    
    ### ===== LOGER ==========        
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (output_dir / 'sop').mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    if args.local_rank == 0 :
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        # -->
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
        if dist_train:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)
        # -->
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    ### ===== DATALOADER ===========
    # -----------------------create dataloader & network & optimizer---------------------------

    def initialize_dataloader(cfg, args, logger, training=True):
        dataset, loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=(args.launcher != 'none'),
            workers=args.workers,
            logger=logger,
            training=training,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs,
        )
        return dataset, loader, sampler

    if args.local_rank == 0 :
        logger.info("Initializing dataset and dataloader...")

    train_set, train_loader, _ = initialize_dataloader(cfg, args, logger, training=True)
    eval_set, eval_loader, _ = initialize_dataloader(cfg, args, logger, training=False)
    
    # Determine subset lengths
    train_len = (
    int(args.dataset_train_len)
    if args.dataset_train_len > 0
    else len(train_set)
    )
    eval_len = (
        int(args.dataset_eval_len)
        if args.dataset_eval_len > 0
        else len(eval_set)
    )
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_set, range(0, train_len))
    eval_subset = torch.utils.data.Subset(eval_set, range(0, eval_len))
    
    # Log dataset information
    print_loader(train_subset, 'train')
    print_loader(eval_subset, 'eval')
        
    ### ========= build Models ===========
    work_path = os.getenv('WORKSF')
    
    model_paths = {
        "git_base": os.path.join(work_path, "datas/transformers/git-base-coco"),
        "git_large": os.path.join(work_path, "datas/transformers/git-large-coco"),
        "blip2": os.path.join(work_path, "datas/transformers/blip2-opt-2.7b"),
        "bert_base": os.path.join(work_path, "datas/transformers/bert-base-uncased"),
    }
    if args.local_rank == 0 :
        logger.info(f"Model paths: {model_paths}")


    ##############################################################################
    ### === Blig2 / GIT ===
    ##############################################################################
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dsi = None
    tokenizer = None
    
    if model_name == "git" or args.git_checkpoint is not None:
        model_dsi_path = args.git_checkpoint if args.git_checkpoint else model_paths["git_base"]
        if args.local_rank == 0 :
            logger.info(f"Initializing GIT model from {model_dsi_path}...")
        
        # Load GIT configuration and model
        config = AutoConfig.from_pretrained(model_dsi_path)
        model_dsi = GitForCausalLM(config).to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_dsi_path)
        
        # Set up lidar model and optionally restore weights
        model = None
        
        model_dsi.set_lidar_model(model, SOP(signed_sqrt=False, do_fc=False), do_use_sop, eval_set.root_path)
        if args.git_checkpoint:
            if args.local_rank == 0 :
                logger.info("Restoring GIT input/output embeddings and lidar parameters...")
            input_embeddings = copy.deepcopy(model_dsi.git.get_input_embeddings())
            output_embeddings = copy.deepcopy(model_dsi.get_output_embeddings())
            bt_norm = copy.deepcopy(model_dsi.lidar_encoder.bt_norm)
            lidar_projection = copy.deepcopy(model_dsi.lidar_encoder.lidar_projection)
            model_dsi.set_input_embeddings(input_embeddings)
            model_dsi.set_output_embeddings(output_embeddings)
            model_dsi.lidar_model.set_lidar_encoder(model, lidar_projection, bt_norm)

    elif model_name == "blip2":
        model_dsi_path = model_paths["blip2"]
        if args.local_rank == 0 :
            logger.info(f"Initializing BLIP2 model from {model_dsi_path}...")
    
        # Load BLIP2 configuration and model
        config = AutoConfig.from_pretrained(model_dsi_path)
        model_dsi = Blip2ModelQuerryLearning(config=config).to(device=device).type(torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_paths["bert_base"])
        
        # Reset BLIP2 parameters if specified
        if args.reset_model:
            if args.local_rank == 0 :
                logger.info("Resetting BLIP2 Q-former and weights...")
            model_dsi.reset_q()
            try:
                model_dsi.qformer.apply(weight_reset)
            except Exception as e:
                logger.error(f"Failed to reset Q-former weights: {traceback.format_exc()}")
        
        # Set lidar model with SOP
        model_sop = SOP(signed_sqrt=False, do_fc=False)
        model_dsi.lidar_model.sop = model_sop
        if args.git_checkpoint:
            if args.local_rank == 0 :
                logger.info("Restoring BLIP2 input/output embeddings and lidar parameters...")
            model_dsi.set_input_embeddings(model_dsi.bert.embeddings, input_embeddings)
            model_dsi.set_output_embeddings(output_embeddings)
            model_dsi.lidar_model.set_lidar_encoder(model, lidar_projection, bt_norm)
        else:
            model_dsi.lidar_model.set_lidar_model(model, model_sop, do_use_sop, eval_set.root_path)
    
    else:
        logger.error(f"Unsupported model name: {model_name}. Must be 'git' or 'blip2'.")


        
    ##############################################################################
    ### === Vocabulary ===
    ##############################################################################

    ### ===== Processor / Tokenizer =====
    processor = AutoProcessor.from_pretrained(model_dsi_path)
    spe_tok = ['[CLS]', '[MASK]', '[PAD]', '[SEP]','[BOS]','[EOS]']
    ukn = tokenizer.convert_tokens_to_ids('[UNK]') # = 100
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token) # ' '
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) # ' '
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) # = 0
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token) # = 102
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token) # = 100
    empt_tk = tokenizer('') # {'input_ids': [101, 102], 'attention_mask': [1, 1]}
    if len(empt_tk.input_ids) == 2 :
        tokenizer.bos_token_id = empt_tk.input_ids[0] # = 101
        tokenizer.eos_token_id = empt_tk.input_ids[1] # = 102
    model_dsi.set_tokenizer(tokenizer,ID_MAX_LENGTH)

    

    ## ==== Vocabulary Filtering / Preprocessing ==== 
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    INT_TOKEN_STR = []
    bad_tk = ['₁','₂','₃','₄','₅','₆','₇','₈','₉','₀','²','¹','³','⁷','⁹','⁰','⁴','⁵','⁶','⁸']
    for token, id in tokenizer.get_vocab().items():
        if token[0] == "#":
            if token[2:].isdigit() and (token[2:] not in bad_tk) :
                INT_TOKEN_IDS.append(id)
                INT_TOKEN_STR.append(token)
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit() and (token[1:] not in bad_tk) :
                INT_TOKEN_IDS.append(id)
                INT_TOKEN_STR.append(token)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
            INT_TOKEN_STR.append(token)
        elif token.isdigit() and (token not in bad_tk) :
            INT_TOKEN_IDS.append(id)
            INT_TOKEN_STR.append(token)
    #INT_TOKEN_IDS.append(tokenizer.bos_token_id)            
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)
    INT_TOKEN_IDS.append(tokenizer.pad_token_id) 
    
    model_dsi.set_vocab(INT_TOKEN_IDS) 
    ############################################################
    # create ID and token lists
    n_subset = [int(x) for x in range(len(eval_subset))] 
    n_set = [int(x) for x in range(len(eval_set))] 
    lid = []
    LIK = []
    for ii in n_subset : lid.append(eval_set.get_label(ii))    
    for ii in lid : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)

    def build_prefix_dict_filter(LIK):
        prefix_dict = {}
        skip_eval_set = 0
        for seq in LIK: # len trainset
            if skip_eval_set % 5 == 0:
                skip_eval_set += 1
                continue
            skip_eval_set += 1
            for sz in range(len(seq) - 1): # length tokens
                prefix = tuple(seq[:sz])  
                next_token = seq[sz]  # The next token
                
                if prefix in prefix_dict:
                    prefix_dict[prefix].add(next_token) 
                else:
                    prefix_dict[prefix] = {next_token}  
        return {k: list(v) for k, v in prefix_dict.items()}  # Convert sets to lists

    
    n_subset = [int(x) for x in range(len(eval_subset))] 
    lid = [eval_set.get_label(ii) for ii in n_subset]
    LIK = [tokenizer(ii, padding="max_length", max_length=ID_MAX_LENGTH).input_ids for ii in lid]
    prefix_dict = build_prefix_dict_filter(LIK)
    
    # Optimized restrict_decode_vocab
    def restrict_decode_vocab_v3(batch_idx, prefix_beam):
        pfb = tuple(prefix_beam.cpu().numpy())  
        return prefix_dict.get(pfb, [102])
    

    
    #update object
    train_set.tokenizer = tokenizer
    train_set.image_processor = processor
    train_set.ID_MAX_LENGTH = ID_MAX_LENGTH
        
    eval_set.tokenizer = tokenizer
    eval_set.image_processor = processor
    eval_set.ID_MAX_LENGTH = ID_MAX_LENGTH


    # enter class indexing collator
    data_collator=IndexingCollator(
        tokenizer,
        padding='longest',
        processor=processor,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 


    logger.info("============== FULL NETWORK STATE =================")
    for name, param in model_dsi.named_parameters() : logger.info(name + "\t =>" + str(param.requires_grad)) 
    logger.info("============== FREE NETWORK STATE =================")

    ### ====== Print active layers git =========
    Print_active_layers_git(args, logger, model_dsi, model_name)  
    logger.info("============== NETWORK STATE =================")

    ### ====== checkpoint =========
    work_path = os.getenv('WORK')
    CHECK_ROOT= work_path + "/checkpoints/"
    checkpoint_dir = CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag

    if not os.path.isdir(checkpoint_dir) :
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    #############################################################################
    ### === Training / Evaluation ===
    ##############################################################################

    if do_train:
        
        train_indices = []  # Indices for the training set
        val_indices = []    # Indices for the validation set
        
        for query_idx in range(train_len): 
            if query_idx % 5 != 0:  # 80% of the data
                train_indices.append(query_idx)
            elif query_idx % 10 != 0:  # 10% for validation
                val_indices.append(query_idx)
            #else:  # Remaining 10% for evaluation
                #eval_indices.append(query_idx)

        # Preload required data for compute metrics only one time 
        def load_json(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        
        label_mapping = {}
        if args.local_rank == 0 :
            print("train_set.labeltype ",train_set.labeltype)
        if train_set.labeltype in {"gps", "hierarchical", "hilbert"}:
            label_mapping = load_json(sequence_path + f"/{train_set.labeltype}.json")
        positions_database = train_set.positions_database # no need to shift --> use to compute metrics
        
        
        train_subset = torch.utils.data.Subset(train_set, train_indices)
        if args.local_rank == 0 :
            print("train_indices ", len(train_indices))
        del train_set, train_indices
        val_subset = torch.utils.data.Subset(eval_set, val_indices)
        if args.local_rank == 0 :
            print("val_indices ", len(val_indices))
        del eval_set, val_indices
        gc.collect()  
        
        # shuffling
        indices = torch.randperm(len(train_subset))
        train_subset_shuffled = torch.utils.data.Subset(train_subset, indices)
       
        if args.local_rank == 0 :
            print("train_subset_shuffled ", len(train_subset))
            print("train_subset_shuffled ", train_subset_shuffled[0]['frame_id'], train_subset_shuffled[1]['frame_id'], train_subset_shuffled[2]['frame_id'])
            print("val_subset ", len(val_subset))
            print("val_subset ", val_subset[0]['frame_id'], val_subset[1]['frame_id'], val_subset[2]['frame_id'])
            

        previous_model_path = args.resume_from_checkpoint


        new_train_batch_size = ori_train_batch_size
        new_eval_batch_size = ori_eval_batch_size
        if args.local_rank == 0 :
            print("new_train_batch_size ", new_train_batch_size)
            print("new_eval_batch_size ", new_eval_batch_size)
        new_training_args = replace(training_args, 
                                    per_device_train_batch_size=new_train_batch_size,
                                    per_device_eval_batch_size=new_eval_batch_size)

        save_file_name = args.save_hit_file # new for naming file hit score

        if args.local_rank == 0 :
            logger.info("  ")
            logger.info(" ======= START TRAINING ========= ")
            logger.info("train_set_len:" + str(len(train_subset)))
            
        trainer = DSITrainer(
                model=model_dsi,
                tokenizer=tokenizer,
                args=new_training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                data_collator=data_collator,
                compute_metrics=make_compute_metrics(tokenizer, logger, args.local_rank, positions_database, label_mapping, save_file_name),
                restrict_decode_vocab=restrict_decode_vocab_v3,
                LIK=LIK,
                id_max_length=ID_MAX_LENGTH
            ) 

        is_trained = False
        if not os.path.isdir(previous_model_path) :
            if args.local_rank == 0 :
                print("train from scratch : ",previous_model_path)
            trainer.train()
            is_trained = True
        else :
            if args.local_rank == 0 :
                print("resume_from_checkpoint : " + previous_model_path)
            trainer.train(resume_from_checkpoint=previous_model_path)
            is_trained = True
        
        if is_trained :
            trainer.save_model(cur_model_path)
            trainer.state.save_to_json(os.path.join(cur_model_path, "trainer_state.json")) 

    if do_eval  : 
        del train_set
        if args.local_rank == 0 :
            print("start eval")
        eval_log3dnet(model_dsi, eval_subset, eval_set, eval_loader, data_collator, tokenizer, cfg, checkpoint_dir, checkp_to_eval, prefix_dict)



if __name__ == '__main__':
    main()






