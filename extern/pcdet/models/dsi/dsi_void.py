from .dsi_template import DSI3DTemplate
from collections import namedtuple    

import numpy as np
import torch.nn as nn
import torch


class DSI_VOID(DSI3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, logger):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
        self.module_list = self.build_networks()
        self.forward_ret_dict = {}
        
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        
        ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict'])


        return batch_dict
        # if self.training:
        #     loss, tb_dict, disp_dict = self.get_training_loss()

        #     ret_dict = {
        #         'loss': loss
        #     }
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
        #     return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        return {}, {}

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        return disp_dict

        

