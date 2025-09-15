import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import os
from module_loader_kitti_pose import *
import math
import json
import argparse



def update_revisits_22( save):
    sequence_path = './LoGG3D-Net/config/kitti_tuples/' 
    
    formatted_data = {}

    f = open(sequence_path + "is_revisit_D-3_T-30.json",) 
    data = json.load(f)
    f.close()
    
    revisit_22 = []
    for i in range(len(list_seq)):
        eval_seq = list_seq[i]
        eval_seq = '%02d' % eval_seq
        print(len(data[eval_seq]))
        for j in range(len(data[eval_seq])):
            revisit_22.append(data[eval_seq][j])
            
    data['22'] = revisit_22
    

    if save:
        with open(sequence_path + "is_revisit_D-3_T-30_v2.json", 'w', encoding ='utf8') as json_file: 
                json.dump(data, json_file, allow_nan=False)
                print("saved dictio", sequence_path + "is_revisit_D-3_T-30.json") 
    



if __name__ == "__main__":
    list_seq = [0, 2, 5, 6, 7, 8]
    parser = argparse.ArgumentParser(description="Hierarchical mapping for KITTI poses")
    parser.add_argument("--eval_seq", type=list, default=list_seq, help="List of sequences used to build 22/")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")
    
    args = parser.parse_args()
    update_revisits_22(args.save) 
