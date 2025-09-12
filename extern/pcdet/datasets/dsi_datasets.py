import pandas as pd
from sklearn.model_selection import train_test_split
import json
import math
from extern.pcdet.datasets.kitti360.kitti360_dataset import load_poses_from_txt, load_timestamps
import numpy as np

from hilbertcurve.hilbertcurve import HilbertCurve

class DSIDatasets:
    def __init__(self,dataset_cfg=None):
        print("ini")
        self.dataset_cfg = dataset_cfg
        self.list_gt = []
        self.dsi_infos_gt = {}
        self.labeltype = 'docid'
        if self.dataset_cfg is not None :    
            self.labeltype =  self.dataset_cfg['LABEL_TYPE']

        self.gpsround = 100
    """
    def load_gt_infos(self,root_path) :
        self.dsi_infos_gt = json.load(open(root_path / "matching.json", "r"))        
    """
    
    def get_hierarchical_label(self,label_id) :
        kk = str(int(label_id))
        if kk in self.hierarchical_label :
            return self.hierarchical_label[kk]
        else :
            print("Warning! key does not exists in hierarchical dict, return classic label")
            return label_id

    def get_gps_label(self,label_id) :
        kk = str(int(label_id))
        print("self.gps_label ", self.gps_label)
        if kk in self.gps_label :
            return self.gps_label[kk]
        else :
            print("Warning! key does not exists in gps dict, return classic label")
            return label_id       


    def label2gps(self,label_id) :
        label_id_gps = self.positions_database[int(label_id)]
        xx = round(label_id_gps[0]*self.gpsround)
        yy = round(label_id_gps[1]*self.gpsround)
        xx_str = f'{xx:05}'
        yy_str = f'{yy:05}'
        res_str = ''.join(x + y for x, y in zip(xx_str, yy_str))
        res_str += xx_str[len(yy_str):] + yy_str[len(xx_str):]
        #import pdb; pdb.set_trace()                    
        return res_str

    def label2hilbert(self,label_id) :
        label_id_gps = self.positions_database[int(label_id)]
        xx = round(label_id_gps[0]*self.gpsround)
        yy = round(label_id_gps[1]*self.gpsround)
    
        p = 17 # Number of iterations (depth of the Hilbert curve) 16 for 06, 17 for 00, 02, 05, 06, 07, 08 and 20 for 22
        n = 2   # Number of dimensions (2D)
        hilbert_curve = HilbertCurve(p, n)
        
        res_str = str(hilbert_curve.distances_from_points([[xx, yy]])[0])
        #import pdb; pdb.set_trace()                    
        return res_str



    def gps2position(self,res_str) :
        xx_str = res_str[::2]
        yy_str = res_str[1::2]
        pos = [float(xx_str)/self.gpsround,float(yy_str)/self.gpsround,0]
        return pos
        # if gpsstr.isnumeric() and len(gpsstr) == 8:
        #     return [float(gpsstr[0:4])/self.gpsround,float(gpsstr[4:8])/self.gpsround,0]
        # else :
        #     return [0,0,0]
        
    
    def get_label_from_path(self,ss) :
        return ss.split('/')[-1][:-4]

    def get_gt_label_from_truth(self,tt,pos=0) :
        n = list(tt.values())[0][0][pos]
        return f'{n:06}'
    
    def get_gt_label(self,index,pos=0) :
        return self.get_gt_label_from_truth(self.dsi_infos_gt[self.get_id(index)],pos) 


    def get_id(self,index): #get label for 4541 bin

        return self.get_label_from_path(self.kitti_infos[index])    

    
    def get_label(self,index): #get label for 4541 bin
    
        ii = self.get_id(index)
        ii = index
        if self.training or self.do_self_eval or True :
            if self.labeltype == 'gps' :
                return self.label2gps(ii)
            elif self.labeltype == 'hierarchical' :
                return self.get_hierarchical_label(ii)
            elif self.labeltype == 'hilbert' :
                return self.label2hilbert(ii)

            elif self.labeltype == 'mixte':
                #return self.label2hilbert(ii) + self.label2gps(ii) + '%06d' %ii + self.get_hierarchical_label(ii)
                return self.label2hilbert(ii) + '%06d' %ii
            
            else : # get label
                return '%06d' %ii
        else :
            return self.get_gt_label(index) 
        
    def filter_dataset(self,nbe) :
        kitti_infos_filtered_train = []
        kitti_infos_filtered_eval = []


        for ii in range(len(self.kitti_infos)) :
            pth = self.kitti_infos[ii]
            iid = self.get_label_from_path(pth)

            if iid in self.dsi_infos_gt :
                kitti_infos_filtered_eval.append(self.kitti_infos[ii])
                gt_lab = self.get_gt_label_from_truth(self.dsi_infos_gt[iid]) 
                self.list_gt.append(gt_lab)

            if len(kitti_infos_filtered_eval) >= nbe :
                break


        for ii in range(len(self.kitti_infos)) :
            pth = self.kitti_infos[ii]
            iid = self.get_label_from_path(pth)
            if iid in self.list_gt or (not self.train_only_revisited)  :
                 kitti_infos_filtered_train.append(self.kitti_infos[ii])


        if self.training :
            nbe_fe=math.floor(len(kitti_infos_filtered_train)/16)*16
            self.kitti_infos = kitti_infos_filtered_train[:nbe_fe]
        else :
            nbe_fe=math.floor(len(kitti_infos_filtered_eval)/16)*16
            self.kitti_infos = kitti_infos_filtered_eval[:nbe_fe]


    def get_dict_dsi(self, index):            
        input_dict = {}
        input_dict['id'] = self.get_id(index)
        input_dict['labels'] = self.get_label(index)

        if self.training or self.do_self_eval  or True  :
            input_dict['gt'] = '-1'
        else :
            input_dict['gt'] = self.get_gt_label(index) 
        return input_dict