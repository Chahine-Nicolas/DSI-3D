#####################################################################################
#####################################################################################
#####################################################################################

import numpy as np
from ..dataset import DatasetTemplate
from ..dsi_datasets import DSIDatasets
import glob
import os
from PIL import Image
import pathlib
import torch
import fnmatch
import random
import re
import json

#####################################################################################
#####################################################################################
#####################################################################################
# Load poses
#####################################################################################

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
#####################################################################################


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
# opti
#####################################################################################
class KittiDataset_2012(DatasetTemplate,DSIDatasets):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        super(DatasetTemplate, self).__init__(dataset_cfg=dataset_cfg)

        self.training = training
        self.kitti_infos = []
        self.eval_seq = int(dataset_cfg['SEQ'])
        self.eval_seq_str = dataset_cfg['SEQ']
        self.root_path = self.root_path / self.eval_seq_str
        self.include_kitti_data()

        # Model components
        self.tokenizer = None
        self.image_processor = None
        self.ID_MAX_LENGTH = None
        self.do_self_eval = dataset_cfg['DO_SELF_EVAL']
        self.train_only_revisited = dataset_cfg['TRAIN_ONLY_REVISITED']

        # Distance Criteria
        self.revisit_criteria = 3
        self.skip_time = 30

        # Labels
        self.hierarchical_label = {}
        self.gps_label = {}

        # Contrastive Learning Parameters
        self._load_contrastive_data(revisit_criteria = 3, revisit_criteria_extended = 3) # change here the negative distance threshold

        # Load dataset files
        self.files = self._load_files()

        # Load poses
        self.positions_database = self._load_positions()

        # Load hierarchical labels
        self.hierarchical_label, self.inv_hierarchical_label = self._load_json_labels('hierarchical_indexing.json')

        # Load GPS labels
        self.gps_label, self.inv_gps_label = self._load_json_labels('dict_gps_2_label_v2.json')

        # Load ground truth info if not self-evaluating
        if not self.do_self_eval:
            self.load_gt_infos(self.root_path)

    def _load_contrastive_data(self, revisit_criteria = 3, revisit_criteria_extended = 20,  skip_time = 0):
        """
        Loads positive and negative sequences for contrastive learning.
        """
        tuple_dir = self.root_path
        kitti_3m_json = f'positive_sequence_D-{revisit_criteria}_T-{skip_time}.json'
        kitti_20m_json = f'positive_sequence_D-{revisit_criteria_extended}_T-{skip_time}.json'

        self.dict_3m = json.load(open(tuple_dir / kitti_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir / kitti_20m_json, "r"))  #define negative distance threshold

    def _load_files(self):
        """
        Loads file information for queries, positives, and negatives.
        """
        fnames = glob.glob(str(self.root_path) + '/velodyne/*.bin')
        inames = sorted(int(os.path.split(fname)[-1][:-4]) for fname in fnames)

        self.kitti_seq_lens = {str(self.eval_seq): len(fnames)}
        files = []

        for query_id in inames:
            positives = self.get_positives(self.eval_seq, query_id)
            negatives = self.get_negatives(self.eval_seq, query_id)
            files.append((self.eval_seq, query_id, positives, negatives))

        return files

    def _load_positions(self):
        """
        Loads pose information from the dataset.
        """
        sequence_path = self.root_path
        _, positions_database = load_poses_from_txt(sequence_path / 'poses.txt')
        # SHIFT MIN TO (0, 0)
        min_bbox = np.min(positions_database, 0)
        return positions_database - min_bbox

    def _load_json_labels(self, filename):
        """
        Loads JSON-based labels such as hierarchical or GPS labels.
        """
        path = self.root_path / filename
        if path.is_file():
            with open(path, "r") as json_file:
                labels = json.load(json_file)
            return labels, {v: k for k, v in labels.items()}
        return {}, {}

    def include_kitti_data(self):
        """
        Loads KITTI dataset files.
        """
        if self.logger:
            self.logger.info('Loading KITTI dataset')

        self.kitti_infos = sorted(
            self.client.list_dir_or_file(self.root_path, list_dir=False, recursive=True, suffix='.bin'),
            key=lambda s: int(re.search(r'\d+', s).group())
        )

        if self.logger:
            self.logger.info(f'Total samples for KITTI dataset: {len(self.kitti_infos)}')

    def get_lidar(self, idx):
        """
        Loads LiDAR data.
        """
        lidar_file = self.root_path / str(idx)
        return self.client.load_to_numpy(str(lidar_file), dtype=np.float32).reshape(-1, 4)


    ###################################################################################################
    ## ajout du chargement des nuages similaire ou non
    ###################################################################################################
    def get_positives(self, sq, index):
        """
        Retrieves positive samples for a given sequence and index.
        Filters out indices that are divisible by 5 to avoid val/eval set contamination.
        """
        sq = str(int(sq))
        index = str(int(index))

        assert sq in self.dict_3m, f"Error: Sequence {sq} not in JSON."

        positives = self.dict_3m[sq].get(index, [])

        # Remove the query index itself from positives
        positives = [x for x in positives if x != int(index) and int(x) % 5 != 0]
        
        return positives

    def get_negatives(self, sq, index):
        """
        Retrieves negative samples for a given sequence and index.
        Ensures negatives are disjoint from positives.
        """
        sq = str(int(sq))
        index = str(int(index))

        assert sq in self.dict_20m, f"Error: Sequence {sq} not in JSON."

        all_ids = set(range(self.kitti_seq_lens[sq]))  # Full range of indices
        neg_set_inv = set(self.dict_20m[sq].get(index, []))  # Indices that should NOT be negatives

        # Get true negatives by subtracting invalid negatives
        negatives = list(all_ids - neg_set_inv)

        # Remove the query index itself & val/eval set contamination
        negatives = [x for x in negatives if x != int(index) and int(x) % 5 != 0]

        return negatives

    """
    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.kitti_seq_lens[str(drive_id)])
        
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        assert len(possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        
        possible_negs = [x for x in possible_negs if int(x) % 5 != 0] # to avoid the use of val / eval set
        
        other_neg_id = random.sample(possible_negs, 1)
        #print("inside other nega", len(possible_negs),  query_id)
        return other_neg_id[0]
    """
    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        """
        Finds an additional negative sample that is dissimilar to all point clouds in the triplet tuple.
        """
        drive_id = str(int(drive_id))
        query_id = str(int(query_id))

        all_ids = set(range(self.kitti_seq_lens[drive_id]))  # All possible indices

        # Collect neighbor IDs from positives and their associated negatives
        neighbour_ids = set(sel_positive_ids)
        for neg in sel_negative_ids:
            neighbour_ids.update(self.get_positives(drive_id, neg))

        # Filter out neighbors to get possible negatives
        possible_negs = list(all_ids - neighbour_ids)
        possible_negs = [x for x in possible_negs if int(x) % 5 != 0]  # Avoid val/eval set

        assert possible_negs, f"No valid negatives for drive {drive_id}, query {query_id}."

        # Randomly select one negative sample
        return random.choice(possible_negs)


    
    ###################################################################################################
    
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

#####################################################################################
#####################################################################################

    # generate matching.json --> have to move
    """
    def preprocess_json(self) :
        #revisit
        revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(int(self.revisit_criteria), int(self.skip_time))

        revisit_json = json.load(open(self.root_path / revisit_json_file, "r"))
        is_revisit_list = revisit_json[self.eval_seq_str]

        self.kitti_infos_revisited = []
        full_dict = {}
        for i in range (self.num_queries):
            dictio = {}
            lidar_path = self.kitti_infos[i]
            query_str = lidar_path.split('/')[-1][:-4]
            query_idx = int(query_str)
            #print(lidar_path," ", query_idx )

            if is_revisit_list[query_idx] == 1.0:
                #print("revisit ")
                self.kitti_infos_revisited.append(lidar_path)
                query_pose = self.positions_database[query_idx]
                query_time = self.timestamps[query_idx]
                d = []
                q = []
                for comp_idx in range(self.num_queries):
                    comp_time = self.timestamps[comp_idx]

                    if (abs(query_time - comp_time) - self.skip_time) < 0:
                            continue
                    if query_idx != comp_idx:
                        comp_pose = self.positions_database[comp_idx]
                        p_dist = np.linalg.norm(query_pose - comp_pose)
                        d.append(round(p_dist,3))
                        q.append(round(comp_idx,3))

                d, q = zip(*sorted(zip(d, q)))
                dictio[query_idx] = [q[:10], d[:10]]
                full_dict[query_str] = dictio
                with open(self.root_path / "json" / (str(query_str) + '.json'), 'w', encoding ='utf8') as json_file: 
                    json.dump(dictio, json_file, allow_nan=False)
        with open(self.root_path / "matching.json", 'w', encoding ='utf8')  as json_file:                     
            json.dump(full_dict, json_file, allow_nan=False)                     
    """


    
    def __getitem__(self, index):
        # Handle merging of iterations to one epoch
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
    
        # Load LiDAR path and extract metadata
        lidar_path = self.kitti_infos[index]
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        path_split = str(lidar_path).split('/')
    
        input_dict = {'frame_id': path_split}

        ### Contrastive
        main_id = int(self.get_id(index))
        drive_id, query_id, positive_ids, negative_ids = self.files[main_id]

        # Select contrastive samples
        sel_positive_ids = random.sample(positive_ids, min(len(positive_ids), 1))
        del positive_ids
        sel_negative_ids = random.sample(negative_ids, min(len(negative_ids), 9))
        del negative_ids
        
        # Get another distinct negative sample
        other_neg_id = self.get_other_negative(drive_id, query_id, sel_positive_ids, sel_negative_ids)
    
        ### Image Processing
        image_path = './extern/proxy.jpg'
        image = Image.open(image_path).convert('RGB')
    
        if self.image_processor:
            input_dict['pixel_values'] = self.image_processor(images=image, return_tensors="pt")['pixel_values'].contiguous()

        ### Load LiDAR Data (if required)
        if "points" in get_item_list:
            input_dict['points'] = self.get_lidar(lidar_path)

            input_dict.update(self.get_dict_dsi(index))  # Merge dictionaries

            # Extract LiDAR values for `lidar_values`
              # Initialize dictionary
            lidar_values = {'batch_size': 1} 
            lidar_values['points'] = torch.tensor(input_dict['points'], dtype=torch.float32) 
            
            
            
    
            # Tokenization (if applicable)
            if self.tokenizer:
                res = self.tokenizer(
                    input_dict['labels'],
                    padding="max_length",
                    return_tensors="pt",
                    truncation='only_first',
                    max_length=self.ID_MAX_LENGTH
                )
    
                input_dict.update({
                    'input_ids': res.input_ids[0],
                    'attention_mask': res.attention_mask[0],
                    'id_pcd_positif':  ['velodyne', f'{sel_positive_ids[0]:06d}.bin'],
                    'id_pcd_negatif':  ['velodyne', f'{sel_negative_ids[0]:06d}.bin'],
                    'other_id_pcd_negatif':  ['velodyne', f'{other_neg_id:06d}.bin']
                })
                
                
    
        # Prepare final data dictionary
        data_dict = self.prepare_data(data_dict=input_dict) # add 'transformation_3d_list': ['random_world_flip', 'random_world_rotation', 'random_world_scaling'], 'transformation_3d_params': {'random_world_flip': [], 'random_world_rotation': 0.0, 'random_world_scaling': 1.0}, 'use_lead_xyz': True, 'transformation_2d_list': [], 'transformation_2d_params': {}}
        
        # Add LIDAR data
        data_dict['points'] = input_dict['points']
        data_dict['lidar_values'] = lidar_values  


        """
         def prepare_lidar_values(self, features):
        lidar_val = {}
        feature_dict = {k: [x[k] for x in features] for k in features[0].keys()}
        
        for key, val in feature_dict.items():
            if key == 'points':
                padded_points = [torch.nn.functional.pad(
                    torch.tensor(coor, dtype=torch.float32),
                    (0, 0, 1, 0), 
                    value=i  
                ) for i, coor in enumerate(val)]
                lidar_val[key] = torch.cat(padded_points, dim=0)
            else:
                lidar_val[key] = np.stack(val, axis=0)  

        return lidar_val
        """ 
        
        desc_from_logg = False 
        desc_from_logg = True
        print("inside")
        if desc_from_logg == True:
            from .logg3d_net_desc import get_logg3d_net_desc
            eval_seq = '06'
            data_dict = get_logg3d_net_desc(data_dict, eval_seq)
        return data_dict
