import numpy as np
from ..dataset import DatasetTemplate
import glob
import os
from PIL import Image
import torch


def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo

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



class Kitti360Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

        self.kitti_infos = []
        self.include_kitti_data()

        self.tokenizer = None
        self.image_processor = None
        self.ID_MAX_LENGTH = None

    def include_kitti_data(self):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        
        self.kitti_infos = list(self.client.list_dir_or_file(
                os.path.join(self.root_path, 'data_3d_raw'),
                list_dir=False, recursive=True, suffix='.bin'
        ))

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))
        

            
    def get_lidar(self, lidar_file):
        return self.client.load_to_numpy(str(self.root_path / 'data_3d_raw' / lidar_file), dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def get_label(self,index):
        return self.kitti_infos[index].split('/')[-1][4:-4]

    
    # def get_seq(self,index) :
    #     return self.kitti_infos[index].split('/')[0]

    # def get_timestamps(self,seq) :
    #     return self.root_path  / 'data_3d_raw' / seq / 'velodyne_points' / 'timestamps.txt'

       
    
    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        print(data_dict['points'])            
        lidar_path = self.kitti_infos[index]
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        
        path_split = str(lidar_path).split('/')
        input_dict = {
            'frame_id': path_split[-4] + '_' + path_split[-1][:-4],
        }
        url = './extern/proxy.jpg'
        image = Image.open(url).convert('RGB')
        #input_dict['truth'] = "-1"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.image_processor is not None : 
            input_dict['pixel_values'] = self.image_processor(images=image, return_tensors="pt")['pixel_values'][:,:,:8,:8].contiguous()
            #import pdb; pdb.set_trace()                   
        if "points" in get_item_list:
            points = self.get_lidar(lidar_path)
            
            input_dict['points'] = points

            input_dict['index'] = self.get_label(index)
            #input_dict['input_text'] = "this is a very nice picture of Paris"
            input_dict['text'] = str(self.get_label(index))

            #import pdb; pdb.set_trace()                   
            #print("input_text :" + input_dict['input_text'])
            if self.tokenizer is not None :
                res = self.tokenizer(input_dict['text'],
                                     padding="max_length",
                                     return_tensors="pt",
                                     truncation='only_first',
                                     max_length=self.ID_MAX_LENGTH)
            
                input_dict['input_ids'] = res.input_ids[0]
                input_dict['attention_mask'] = res.attention_mask[0]


            #print("WARNING : stratégie naive de vocab, les index change en fonction des iters")
            input_dict['labels'] = input_dict['text']

                 
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['points'] = input_dict['points']
        return data_dict



