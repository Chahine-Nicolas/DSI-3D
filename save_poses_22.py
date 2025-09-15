import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import os
from module_loader_kitti_pose import *
import math
import json
import argparse



def save_poses_22(kitti_dir, save):
    ud_content = []
    
    for i in range(len(list_seq)):
        eval_seq = list_seq[i]
    

        eval_seq = '%02d' % eval_seq
        print("eval_seq ", eval_seq)
        sequence_path = kitti_dir + 'sequences/' + eval_seq + '/'
        file = open(sequence_path + "poses.txt", "r")
        
        content=file.readlines()
        print(content[-1])
        file.close()
    
        tfs, pose = load_poses_from_txt(sequence_path + 'poses.txt')
    
        min_bbox = np.min(pose,0) 
        print("min_bbox  ",min_bbox )
    
        
        data = content
        shift = i*1000
        updated_strings = []
        for row in data:
            # Split the string into a list of numbers
            numbers = row.split()
            # Convert the last three elements to floats, add 1000, and format back to scientific notation
        
            numbers[3] = str(float(numbers[3]) + shift - min_bbox[0] )
            numbers[7] = str(float(numbers[7]) + shift - min_bbox[2])
            numbers[11] = str(float(numbers[11]) + shift - min_bbox[1])
            # Join the updated numbers back into a single string
            updated_strings.append(" ".join(numbers))
        ud_content.append(updated_strings)
    
    if save:
        with open( kitti_dir + 'sequences/22/poses.txt', 'w') as f:
            for seq in range(len(list_seq)):
                for line in ud_content[seq]:
                    f.write("%s\n" % line)
                    #print(line)
        print("saved at ", kitti_dir + 'sequences/22/poses.txt') 

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical mapping for KITTI poses")
    
    kitti_dir = os.getenv("WORKSF") + "/datas/datasets/"  
    parser.add_argument("--data_path", type=str,  default=kitti_dir, help="Dataset path")
    
    list_seq = [0, 2, 5, 6, 7, 8]
    parser.add_argument("--eval_seq", type=list, default=list_seq, help="List of sequences used to build 22/")
    parser.add_argument("--save", type=bool, default=False, help="Save result as JSON")
    
    args = parser.parse_args()
    save_poses_22(args.data_path, args.save) 


