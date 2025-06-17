from .dsi_template import DSI3DTemplate
from collections import namedtuple    

import numpy as np
import torch.nn as nn
import torch

class DSI(DSI3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}
        
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        batch_size= batch_dict['batch_size']
        return batch_dict

        
    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        return disp_dict



