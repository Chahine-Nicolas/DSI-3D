import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import os
from module_loader_kitti_pose import *
import math
import json
import argparse


def rename_revisit(data):
    list_seq = [0, 2, 5, 6, 7, 8]
    
    sum_id = 0
    formatted_data = {}
    for seq, indices in data.items():
        if not int(seq) in list_seq:
            continue
            
        print(seq, sum_id)
        # Format the sequence key
        formatted_seq = seq
        formatted_data[formatted_seq] = {}
        
        for idx, values in indices.items():
            # Format the index key and combine it with the formatted sequence
            formatted_idx = '%06d' % ( int(idx) + sum_id ) 
            # Format each value in the list to also include the sequence
            formatted_values = ['%06d' % ( int(v) + sum_id) for v in values]
            # Add to the new dictionary
            formatted_data[formatted_seq][formatted_idx] = formatted_values
        sum_id += len(data[str(seq)])
    return formatted_data


def save_positive_per_seq(kitti_dir, save):

     sequence_path = kitti_dir + "/sequences/22/"
    
    # rename and save positive_sequence_D-3_T-0.json
    f = open(kitti_dir + "/sequences/00/positive_sequence_D-3_T-0.json") 
    data = json.load(f)
    f.close()
    formatted_data = rename_revisit(data)
    if save:
        with open(sequence_path + "positive_sequence_D-3_T-0.json", 'w', encoding ='utf8') as json_file: 
            json.dump(formatted_data, json_file, allow_nan=False) 
        print("saved dictio", sequence_path + "positive_sequence_D-3_T-0.json")



    # rename and save positive_sequence_D-20_T-0.json 
    f = open(kitti_dir + "/sequences/00/positive_sequence_D-20_T-0.json") 
    data = json.load(f)
    f.close()
    formatted_data = rename_revisit(data)
    if save:
        with open(sequence_path + "positive_sequence_D-20_T-0.json", 'w', encoding ='utf8') as json_file: 
            json.dump(formatted_data, json_file, allow_nan=False) 
        print("saved dictio", sequence_path + "positive_sequence_D-20_T-0.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical mapping for KITTI poses")
    
    kitti_dir = os.getenv("WORKSF") + "/datas/datasets/"  
    parser.add_argument("--data_path", type=str,  default=kitti_dir, help="Dataset path")
    
    list_seq = [0, 2, 5, 6, 7, 8]
    parser.add_argument("--eval_seq", type=list, default=list_seq, help="List of sequences used to build 22/")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")
    
    args = parser.parse_args()
    
    save_positive_per_seq(args.data_path, args.save) 

# python save_positive_per_seq.py 

