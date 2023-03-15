import numpy as np
import os
import torch
import math
import random
import pickle
import json

class BaseParams:
    exp_id = '0'
    manual_seed = random.randint(1, 10000)  
    use_cuda = True
    max_epoch = 200000
    save_checkpoint_iter = 200
    validate_iter = 100
    batch_size = 16
    lr = 1e-3
    root_file = './record'
    load_core = None
    def __init__(self): 
        if (self.use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
        self.summary_dir = os.path.join(self.root_file, 'log', self.exp_id)
        self.checkpoint_dir = os.path.join(self.root_file, 'state', self.exp_id)
        print('exp_id: ', self.exp_id)
        if self.load_core is not None:
            with open(self.load_core, 'r') as fp:
                self.core = json.load(fp)
        
class ExpParams(BaseParams):
    def __init__(self):
        self.exp_id = 'test'
        self.core = {
            'data':{    'BaseFile':  None,
                        'load_checkpoint_dir': None,
                        'DataDir': "./data/1021_1/",
                        '1_res_dir': None,
                        '2_res_dir': None,
            }
        }
        if self.core['data']['1_res_dir'] is None:
            self.core['data']['1_res_dir'] = self.core['data']['DataDir'] + '1_res_m.mat'
        if self.core['data']['2_res_dir'] is None:
            self.core['data']['2_res_dir'] = self.core['data']['DataDir'] + '2_res_m.mat'
        super().__init__()