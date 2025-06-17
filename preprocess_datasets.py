### GD-MAE
import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path

from extern.pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from extern.pcdet.datasets import build_dataloader
from extern.pcdet.models import build_network, model_fn_decorator
from extern.pcdet.utils import common_utils
from extern.pcdet.datasets.kitti360.kitti360_dataset import load_poses_from_txt, load_timestamps

from scipy.spatial.distance import cdist
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,required=True, help='specify the config for training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'


    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


# python -i main_process_dataset.py --launcher none --cfg_file ./config.yaml
def main():
    args, cfg = parse_config()

    ## GD-MAE parser
    batch_size = 1
    dist_train = False
    LOCAL_RANK = 0
    epoch=1
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_train, workers=args.workers,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=epoch
    )

    train_set.preprocess_json()
    
    seq = -1
    for ii in range(0,10) :
        print((train_set.root_path / 'data_3d_raw' / train_set.kitti_infos[ii]))
        seq = train_set.get_seq(ii)
        print(seq)

    # ts = train_set.get_timestamps(seq)        
    # timestamps = load_timestamps(ts)
    # import pdb; pdb.set_trace()    
        
    


    
if __name__ == '__main__':
    main()
 #   main_gd_mae()






