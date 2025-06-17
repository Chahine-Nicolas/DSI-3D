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
from torch.nn.parallel import DistributedDataParallel as DDP

import hostlist

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter 
import copy 
import traceback
import logging

from extern.pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from extern.pcdet.datasets import build_dataloader
from extern.pcdet.models import build_network, model_fn_decorator
from extern.pcdet.utils import common_utils
from extern.train_utils.optimization import build_optimizer, build_scheduler
from extern.train_utils.train_utils import train_model
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
##
from evaluate_gate import eval_log3dnet
#from evaluate_overfit import eval_overfit
from compute_hierarchical_index import compute_hierarchical_clustering

import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import TrainerCallback

##################################
# read pos
# #################################
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
    
    parser.add_argument('--extra_tag_1', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_2', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_3', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_4', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_5', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--extra_tag_6', type=str, default='default', help='extra tag for this experiment')
    
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', type=int, default=-1, help='seed')    
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
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

    parser.add_argument('--save_hit_file',  type=str, default='hit.txt', help='file with hit score')

    parser.add_argument('--id_max_length', type=int, default=10, required=False, help='adam_epsilon')

    parser.add_argument('--eval_chkt_1', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--eval_chkt_2', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--eval_chkt_3', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--eval_chkt_4', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--eval_chkt_5', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')
    parser.add_argument('--eval_chkt_6', type=str, default="checkpoint-100", required=False, help='checkpoint to be evaluated')

    parser.add_argument('--lr_scheduler_type', type=str, default="linear", required=False, help='LR scheduler type')

    parser.add_argument('--num_cycles', type=int, default=1, required=False, help='restart cycles')

    parser.add_argument('--encoder_chkp', type=str, default=None, required=False, help='checkpoint of the 3D encoder')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.sync_bn = args.sync_bn or cfg.OPTIMIZATION.get('SYNC_BN', False)
    

    return args, cfg


##############################################################################
# Optimisé make_compute_metrics
# #############################################################################
def make_compute_metrics(tokenizer, logger, rank, positions_database, label_mapping, save_file_name):
#def make_compute_metrics(tokenizer, logger, rank, train_set, sequence_path, d=None):
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

        #hit_at_1_tensor = torch.tensor(hit_at_1, device="cuda")
        #hit_at_10_tensor = torch.tensor(hit_at_10, device="cuda")
        #dist.all_reduce(hit_at_1_tensor, op=dist.ReduceOp.SUM)
        #dist.all_reduce(hit_at_10_tensor, op=dist.ReduceOp.SUM)

        total_predictions = len(eval_preds.predictions)

        #######################################################################
        # save metrics
        #######################################################################
        with open(save_file_name, 'a') as f:
            f.write(str(hit_at_1 / total_predictions ) + " " + str(hit_at_10 / total_predictions ) + "\n")
        #f.close()
        #######################################################################
        #######################################################################
        
        return {
            "Hits@1": hit_at_1 / total_predictions,
            "Hits@10": hit_at_10 / total_predictions,
        }
    
    return compute_metrics

##############################################################################
# #############################################################################

##############################################################################
# Optimisé DSITrainer
# #############################################################################
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
        #model.half()
        
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
            #scores = batch_beams_dict['scores']
            

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
##############################################################################
# #############################################################################

@dataclass
##############################################################################
# Optimisé IndexingCollator
# #############################################################################
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



def load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set):
    if model_name == "git" or args.git_checkpoint is not None:
        #model_dsi_path = args.git_checkpoint if args.git_checkpoint else model_paths["git_base"]
        model_dsi_path = args.git_checkpoint if args.git_checkpoint else model_paths["git_base"]
        if args.local_rank == 0 :
            logger.info(f"Initializing GIT model from {model_dsi_path}...")
        
        # Load GIT configuration and model
        config = AutoConfig.from_pretrained(model_dsi_path)
        model_dsi = GitForCausalLM(config).to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_dsi_path)
        
        # Set up lidar model and optionally restore weights
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
    return model_dsi, tokenizer



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

    """
    checkp_to_eval_1 = args.eval_chkt_1
    checkp_to_eval_2 = args.eval_chkt_2
    checkp_to_eval_3 = args.eval_chkt_3
    checkp_to_eval_4 = args.eval_chkt_4
    checkp_to_eval_5 = args.eval_chkt_5
    checkp_to_eval_6 = args.eval_chkt_6

    checkp_to_eval = [checkp_to_eval_1, checkp_to_eval_2, checkp_to_eval_3, checkp_to_eval_4, checkp_to_eval_5, checkp_to_eval_6]
    """
    expert_checkp = {
        0: args.eval_chkt_1, #00,
        1: args.eval_chkt_2, #02,
        2: args.eval_chkt_3, #05,
        3: args.eval_chkt_4, #06,
        4: args.eval_chkt_5, #07,
        5: args.eval_chkt_6 #08
    }


    encoder_checkp = args.encoder_chkp
    
    do_overfit = True
    random_seed = int(args.fix_random_seed)
    do_use_sop = eval(args.use_sop)
    if args.launcher == "pytorch" : 
        args.local_rank = int(os.environ['LOCAL_RANK'])


    sequence_path = cfg['DATA_CONFIG']['DATA_PATH'] + cfg['DATA_CONFIG']['SEQ']
    
    ##########################################################################################
    save_file_name = args.save_hit_file # new for naming file hit score
    with open(save_file_name, 'w') as f:
        print(' pour créer / vider le txt')
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

    #import pdb; pdb.set_trace()    
    ## GD-MAE parser
    #training_args.train_batch_size = batch_size
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
    
    #import pdb; pdb.set_trace()
    
    batch_size = training_args.per_device_train_batch_size
    ori_train_batch_size  = training_args.per_device_train_batch_size
    ori_eval_batch_size  = training_args.per_device_eval_batch_size
    args.batch_size = batch_size

    
    ### ===== LOGER ==========        
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag_1
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

    #train_set, train_loader, _ = initialize_dataloader(cfg, args, logger, training=True)
    #eval_set, eval_loader, _ = initialize_dataloader(cfg, args, logger, training=False)

    if args.local_rank == 0 :
        logger.info('Load seq by seq')
    original_cfg_DATA_CONFIG_SEQ = cfg['DATA_CONFIG']['SEQ'] 
    cfg['DATA_CONFIG']['SEQ'] = '00'
    eval_set_0, eval_loader_0, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = '02'
    eval_set_1, _, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = '05'
    eval_set_2, _, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = '06'
    eval_set_3, _, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = '07'
    eval_set_4, _, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = '08'
    eval_set_5, _, _ = initialize_dataloader(cfg, args, logger, training=False)
    cfg['DATA_CONFIG']['SEQ'] = original_cfg_DATA_CONFIG_SEQ

    eval_set, eval_loader = eval_set_0, eval_loader_0

    expert_hierar_label = {
        0: eval_set_0.get_hierarchical_label, #00,
        1: eval_set_1.get_hierarchical_label, #02,
        2: eval_set_2.get_hierarchical_label, #05,
        3: eval_set_3.get_hierarchical_label, #06,
        4: eval_set_4.get_hierarchical_label, #07,
        5: eval_set_5.get_hierarchical_label #08
    }

    
    # Determine subset lengths
    """
    train_len = (
    int(args.dataset_train_len)
    if args.dataset_train_len > 0
    else len(train_set)
    )"""
    eval_len = (
        int(args.dataset_eval_len)
        if args.dataset_eval_len > 0
        else len(eval_set)
    )
    
    # Create subsets
    #train_subset = torch.utils.data.Subset(train_set, range(0, train_len))
    eval_subset = torch.utils.data.Subset(eval_set, range(0, eval_len))

    eval_subset_0 = torch.utils.data.Subset(eval_set_0, range(0, 4541))
    eval_subset_1 = torch.utils.data.Subset(eval_set_1, range(0, 4661))
    eval_subset_2 = torch.utils.data.Subset(eval_set_2, range(0, 2761))
    eval_subset_3 = torch.utils.data.Subset(eval_set_3, range(0, 1101))
    eval_subset_4 = torch.utils.data.Subset(eval_set_4, range(0, 1101))
    eval_subset_5 = torch.utils.data.Subset(eval_set_5, range(0, 4071))



    


    
    
    # Log dataset information
    #print_loader(train_subset, 'train')
    print_loader(eval_subset, 'eval')
        
    ### ========= build Models ===========
    work_path = os.getenv('WORKSF')
    #/lustre/fsn1/worksf/projects/rech/dki/ujo91el/datas/transformers

    model_paths = {
        "git_base": os.path.join(work_path, "datas/transformers/git-base-coco"),
        "git_large": os.path.join(work_path, "datas/transformers/git-large-coco"),
        "blip2": os.path.join(work_path, "datas/transformers/blip2-opt-2.7b"),
        "bert_base": os.path.join(work_path, "datas/transformers/bert-base-uncased"),
    }
    if args.local_rank == 0 :
        logger.info(f"Model paths: {model_paths}")
    
    # Build model
    num_class = len(cfg.CLASS_NAMES)
    if args.local_rank == 0 :
        logger.info(f"Building model with {num_class} classes.")
    model = build_network(model_cfg=cfg.MODEL, num_class=num_class, dataset=eval_set, logger=logger)

    # Apply SyncBatchNorm if required
    if args.sync_bn:
        logger.info("Converting to SyncBatchNorm...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Move model to device
    model = model.to(device)
    if args.local_rank == 0 :
        logger.info(f"Model moved to {device}.")

    # Initialize optimizer
    if args.local_rank == 0 :
        logger.info("Building optimizer...")
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)


    #import pdb; pdb.set_trace()
    #from accelerate import Accelerator
    #accelerator = Accelerator()
    #train_dl, eval_dl, model, optimizer = accelerator.prepare(train_loader, eval_loader, model, optimizer)
    #train_dl, eval_dl, model, optimizer = accelerator.prepare(train_set, eval_set, model, optimizer)
    #import pdb; pdb.set_trace()

    
    # Load checkpoints
    start_epoch = 0
    it = 0
    last_epoch = -1
    if args.pretrained_model:
        if args.local_rank == 0 :
            logger.info(f"Loading pretrained model from {args.pretrained_model}...")
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
    elif args.ckpt:
        if args.local_rank == 0 :
            logger.info(f"Loading model and optimizer state from {args.ckpt}...")
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
        )
        last_epoch = start_epoch + 1
    else:
        # Load the most recent checkpoint in the directory if available
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if ckpt_list:
            ckpt_list.sort(key=os.path.getmtime)
            latest_ckpt = ckpt_list[-1]
            if args.local_rank == 0 :
                logger.info(f"Resuming from the latest checkpoint: {latest_ckpt}")
            it, start_epoch = model.load_params_with_optimizer(
                latest_ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    # Set model to training mode
    model.train()
    if args.local_rank == 0 :
        logger.info("Model is set to training mode.")


    ##############################################################################
    ### === Blig2 / GIT ===
    ##############################################################################
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dsi = None
    tokenizer = None

    device2 = 'cuda:1'

    model_dsi_expert_1, tokenizer_1 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_0)
    model_dsi_expert_2, tokenizer_2 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_1)
    model_dsi_expert_3, tokenizer_3 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_2)
    model_dsi_expert_4, tokenizer_4 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_3)
    model_dsi_expert_5, tokenizer_5 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_4)
    model_dsi_expert_6, tokenizer_6 = load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set_5)



    """
    torch.cuda.device_count()
    devices = [device1, device2, device3, device4, device5, device6]

    models_dsi_expert = [
        load_model(args, model_name, model_paths, logger, device, model, do_use_sop, eval_set)
        for device in devices
    ]"""
    model_dsi_path = args.git_checkpoint if args.git_checkpoint else model_paths["git_base"]
    ##############################################################################
    ##############################################################################

    ### ===== Processor / Tokenizer =====


    def set_models_tokenizers(model, tokenizer):
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
            
        model.set_tokenizer(tokenizer,ID_MAX_LENGTH)
        return model, processor, tokenizer

    model_dsi_expert_1,processor_1, tokenizer_1 = set_models_tokenizers(model_dsi_expert_1, tokenizer_1)
    model_dsi_expert_2,processor_2, tokenizer_2 = set_models_tokenizers(model_dsi_expert_2, tokenizer_2)
    model_dsi_expert_3,processor_3, tokenizer_3  = set_models_tokenizers(model_dsi_expert_3, tokenizer_3)
    model_dsi_expert_4,processor_4, tokenizer_4 = set_models_tokenizers(model_dsi_expert_4, tokenizer_4)
    model_dsi_expert_5,processor_5, tokenizer_5 = set_models_tokenizers(model_dsi_expert_5, tokenizer_5)
    model_dsi_expert_6,processor_6, tokenizer_6 = set_models_tokenizers(model_dsi_expert_6, tokenizer_6)

    expert_tokenizer = {
        0: tokenizer_1, #00,
        1: tokenizer_2, #02,
        2: tokenizer_3, #05,
        3: tokenizer_4, #06,
        4: tokenizer_5, #07,
        5: tokenizer_6 #08
    }
    ############################################################
    
    ## ==== Vocabulary Filtering / Preprocessing ==== 

    def set_models_vocab(model, tokenizer):
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
    
        model.set_vocab(INT_TOKEN_IDS) 
        return model

    model_dsi_expert_1 = set_models_vocab(model_dsi_expert_1, tokenizer_1)
    model_dsi_expert_2 = set_models_vocab(model_dsi_expert_2, tokenizer_2)
    model_dsi_expert_3 = set_models_vocab(model_dsi_expert_3, tokenizer_3)
    model_dsi_expert_4 = set_models_vocab(model_dsi_expert_4, tokenizer_4)
    model_dsi_expert_5 = set_models_vocab(model_dsi_expert_5, tokenizer_5)
    model_dsi_expert_6 = set_models_vocab(model_dsi_expert_6, tokenizer_6)

    

    ############################################################
    # create ID and token lists
    """
    n_subset = [int(x) for x in range(len(eval_subset))] 
    n_set = [int(x) for x in range(len(eval_set))] 
    lid = []
    LIK = []
    for ii in n_subset : lid.append(eval_set.get_label(ii))    
    for ii in lid : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
    """

    def make_token_list(tokenizer, processor, len_eval_subset, eval_set):
        n_subset = [int(x) for x in range(len_eval_subset)] 
        lid = []
        LIK = []
        for ii in n_subset : lid.append(eval_set.get_label(ii))  
        #for ii in n_subset : print(ii, eval_set.get_label(ii))
        #import pdb; pdb.set_trace()
        for ii in lid : LIK.append(tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH).input_ids)
        #for ii in lid : print(ii, tokenizer(ii,padding="max_length",max_length=ID_MAX_LENGTH))    
        return LIK
        
    LIK_1 = make_token_list(tokenizer_1, processor_1, 4541, eval_set_0)
    #LIK_1.sort()
    #print("LIK_1[:10]")
    #import pdb; pdb.set_trace()
    LIK_2 = make_token_list(tokenizer_2, processor_2, 4661, eval_set_1)
    LIK_3 = make_token_list(tokenizer_3, processor_3, 2761, eval_set_2)
    LIK_4 = make_token_list(tokenizer_4, processor_4, 1101, eval_set_3)
    LIK_5 = make_token_list(tokenizer_5, processor_5, 1101, eval_set_4)
    LIK_6 = make_token_list(tokenizer_6, processor_6, 4071, eval_set_5)


    expert_LIK = {
        0: LIK_1, #00,
        1: LIK_2, #02,
        2: LIK_3, #05,
        3: LIK_4, #06,
        4: LIK_5, #07,
        5: LIK_6 #08
    }
    
    
    #shorter token list
    #INT_TOKEN_IDS = sorted(set(np.array(LIK).flatten()))
    #model_dsi.set_vocab(INT_TOKEN_IDS) 
    
    #tokenizer.decode(, skip_special_tokens=True)
    #import pdb; pdb.set_trace()
    
    def restrict_decode_vocab(batch_idx, prefix_beam):
        TOK_ID_OK = []
        sz = len(prefix_beam)
        pfb = prefix_beam.cpu().numpy()
        #import pdb; pdb.set_trace()

        for tt in LIK :
            #print("tt[:sz] ",tt[:sz], " pfb.tolist() ", pfb.tolist())
            if tt[:sz] == pfb.tolist()  :
                TOK_ID_OK.append(tt[sz])
        #print("tok:" + str(TOK_ID_OK))
        if len(TOK_ID_OK) == 0 :
            TOK_ID_OK.append(102)
        return TOK_ID_OK
    ############################################################

    ############################################################

    """
    # Build Prefix Lookup Dictionary for O(1) Lookup
    def build_prefix_dict(LIK, tokenizer):
        prefix_dict = {}
        for seq in LIK: # len trainset
            #tokenizer.decode(seq, skip_special_tokens=True)
            #import pdb; pdb.set_trace()
            for sz in range(len(seq) - 1): # length tokens
                prefix = tuple(seq[:sz])  
                next_token = seq[sz]  # The next token
                
                if prefix in prefix_dict:
                    prefix_dict[prefix].add(next_token) 
                else:
                    prefix_dict[prefix] = {next_token}  
        return {k: list(v) for k, v in prefix_dict.items()}  # Convert sets to lists
    """

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
    #n_subset = n_subset[:4541] #00
    #n_subset = n_subset[9201:11962] #05
    #n_subset = n_subset[11962:13063] #06
    #n_subset = n_subset[13063:14164] #07
    # n_subset = n_subset[14164:] #08

    
    #lid = [eval_set.get_label(ii) for ii in n_subset] 
    #LIK = [tokenizer(ii, padding="max_length", max_length=ID_MAX_LENGTH).input_ids for ii in lid]
    
    #import pdb; pdb.set_trace()
    #prefix_dict = build_prefix_dict(LIK, tokenizer) # new
    
    #prefix_dict = build_prefix_dict_filter(LIK)

    """
    prefix_dict_1 = build_prefix_dict_filter(LIK_1)
    prefix_dict_2 = build_prefix_dict_filter(LIK_2)
    prefix_dict_3 = build_prefix_dict_filter(LIK_3)
    prefix_dict_4 = build_prefix_dict_filter(LIK_4)
    prefix_dict_5 = build_prefix_dict_filter(LIK_5)
    prefix_dict_6 = build_prefix_dict_filter(LIK_6)
    """
    expert_prefix = {
        0: build_prefix_dict_filter(LIK_1), #00,
        1: build_prefix_dict_filter(LIK_2), #02,
        2: build_prefix_dict_filter(LIK_3), #05,
        3: build_prefix_dict_filter(LIK_4), #06,
        4: build_prefix_dict_filter(LIK_5), #07,
        5: build_prefix_dict_filter(LIK_6) #08
    }

    
    #import pdb; pdb.set_trace()
    # Optimized restrict_decode_vocab
    def restrict_decode_vocab_v3(batch_idx, prefix_beam):
        pfb = tuple(prefix_beam.cpu().numpy())  
        return prefix_dict.get(pfb, [102])
    
    #sprefix_dict = []
    ############################################################
    
    # restrict code version DSI
    #def restrict_decode_vocab_v2(batch_idx, prefix_beam): #
        #return INT_TOKEN_IDS
    

    #update object
    #train_set.tokenizer = tokenizer_1
    #train_set.image_processor = processor_1
    #train_set.ID_MAX_LENGTH = ID_MAX_LENGTH

    """
    print(eval_set[0].keys())
    print(eval_set_1[0].keys())
    print(eval_subset[0].keys())
    print(eval_subset_1[0].keys())
    import pdb; pdb.set_trace()
    """
    
    eval_set.tokenizer = tokenizer_1
    eval_set.image_processor = processor_1
    eval_set.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_0.tokenizer = tokenizer_1
    eval_set_0.image_processor = processor_1
    eval_set_0.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_1.tokenizer = tokenizer_2
    eval_set_1.image_processor = processor_2
    eval_set_1.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_2.tokenizer = tokenizer_3
    eval_set_2.image_processor = processor_3
    eval_set_2.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_3.tokenizer = tokenizer_4
    eval_set_3.image_processor = processor_4
    eval_set_3.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_4.tokenizer = tokenizer_5
    eval_set_4.image_processor = processor_5
    eval_set_4.ID_MAX_LENGTH = ID_MAX_LENGTH

    eval_set_5.tokenizer = tokenizer_6
    eval_set_5.image_processor = processor_6
    eval_set_5.ID_MAX_LENGTH = ID_MAX_LENGTH

    """
    print(eval_set[0].keys())
    print(eval_set_1[0].keys())
    print(eval_subset[0].keys())
    print(eval_subset_1[0].keys())
    import pdb; pdb.set_trace()
    """
    
    expert_eval_subset = {
        0: eval_subset_0, #00,
        1: eval_subset_1, #02,
        2: eval_subset_2, #05,
        3: eval_subset_3, #06,
        4: eval_subset_4, #07,
        5: eval_subset_5  #08
    }

    # enter class indexing collator
    data_collator_1=IndexingCollator(
        tokenizer_1,
        padding='longest',
        processor=processor_1,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 
    
    data_collator_2=IndexingCollator(
        tokenizer_2,
        padding='longest',
        processor=processor_2,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    data_collator_3=IndexingCollator(
        tokenizer_3,
        padding='longest',
        processor=processor_3,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    data_collator_4=IndexingCollator(
        tokenizer_4,
        padding='longest',
        processor=processor_4,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    data_collator_5=IndexingCollator(
        tokenizer_5,
        padding='longest',
        processor=processor_5,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    data_collator_6=IndexingCollator(
        tokenizer_6,
        padding='longest',
        processor=processor_6,
        id_max_length=ID_MAX_LENGTH,
        batch_size=args.batch_size) # = dict with 

    """
    print(eval_set[0].keys())
    print(eval_set_1[0].keys())
    import pdb; pdb.set_trace()
    
    data_collator_1(torch.utils.data.Subset(eval_subset_0,range(0,1)))  
    data_collator_2(torch.utils.data.Subset(eval_subset_1,range(0,1)))  

    data_collator_3(torch.utils.data.Subset(eval_subset_2,range(0,1)))  
    data_collator_4(torch.utils.data.Subset(eval_subset_3,range(0,1)))  

    data_collator_5(torch.utils.data.Subset(eval_subset_4,range(0,1)))  
    data_collator_6(torch.utils.data.Subset(eval_subset_5,range(0,1)))  
    """
    
    expert_data_collator = {
        0: data_collator_1, #00,
        1: data_collator_2, #02,
        2: data_collator_3, #05,
        3: data_collator_4, #06,
        4: data_collator_5, #07,
        5: data_collator_6 #08
    }
    
    ### ====== Freezing Model =========
    ## Freeze network
    model.freeze(model.model_cfg.FREEZE_LAYERS) # lidar_model
    if True : 
        if args.local_rank == 0 :
            logger.info("============== FULL NETWORK STATE =================")
        for name, param in model_dsi_expert_1.named_parameters() : logger.info(name + "\t =>" + str(param.requires_grad)) 
    if args.local_rank == 0 :
        logger.info("============== FREE NETWORK STATE =================")

    ### ====== Print active layers git =========
    if args.local_rank == 0 :
        Print_active_layers_git(args, logger, model_dsi_expert_1, model_name)  
        logger.info("============== NETWORK STATE =================")

    ### ====== checkpoint =========
    work_path = os.getenv('WORK')
    CHECK_ROOT= work_path + "/checkpoints/"
    #checkpoint_dir = CHECK_ROOT + "/" + git + "_" + 'hierarchical' + "_" + args.extra_tag

    checkpoint_dir = {
        0: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_1, #00,
        1: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_2, #02,
        2: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_3, #05,
        3: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_4, #06,
        4: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_5, #07,
        5: CHECK_ROOT + "/" + model_name + "_" + eval_set.labeltype + "_" + args.extra_tag_6 #08
    }

    
    #checkpoint_dir = CHECK_ROOT + eval_set.labeltype + "_" + args.extra_tag
    if not os.path.isdir(checkpoint_dir[0]) :
        Path(checkpoint_dir[0]).mkdir(parents=True, exist_ok=True)


    #if do_train:

    if do_eval  : 
        #del train_set
        if args.local_rank == 0 :
            print("start eval")
            
        eval_log3dnet(model_dsi_expert_1, model_dsi_expert_2,model_dsi_expert_3,model_dsi_expert_4,model_dsi_expert_5,model_dsi_expert_6, eval_subset, eval_set, eval_loader, expert_data_collator, tokenizer, cfg, checkpoint_dir, expert_checkp, expert_tokenizer, expert_prefix, expert_hierar_label, expert_eval_subset, encoder_checkp)

    """ # in separate file
    if do_preprocess  :
        compute_hierarchical_clustering(train_subset,train_set,data_collator,tokenizer,cfg)        
        model_dsi.eval()
        eval_log3dnet(model_dsi, eval_subset, eval_set, data_collator, tokenizer, cfg)
    """

if __name__ == '__main__':
    main()






