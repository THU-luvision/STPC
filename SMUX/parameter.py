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
    save_checkpoint_iter = 100
    validate_iter = 20
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
        
class SimParams(BaseParams):
    def __init__(self):
        self.exp_id = 'test'
        self.core = {
            'network':{ 'ExpMode': 0, 'input_size': 2, 'output_size': 4, 'frame_num': 5,
                        'hidden_size': 5, 'output_dim': 8,
                        'NoiseStd': [0, 0, 0], 'Scale':[1, 2, 50], 
                        'Pro_kw':{ 'SigK': 1}
                        },
            'data':{    'BaseFile': "**.pth.tar",
                        'PropagationFile': "**.pth.tar",
                        'load_checkpoint_dir': None,
                        'SOAFile':  "**.mat",
                        'CoeffFile': "**.mat"
            }
        }
        self.loss = {
            'regu': 1000, 'thers':  0.17, 'sp': 0, 'regu2': 40, 'regu3':5
        }
        self.lr = [1e-2, 1e-4] 
        self.SigK_iter = 10
        self.SigK_Sq = iter([1, 10, 100, None])
        self.SigK_iter_Sq = iter([1, 10, 10, 10])
        if self.SigK_iter is not None:
            self.save_checkpoint_iter = 10
        super().__init__()
        
class ExpParams(BaseParams):
    def __init__(self):
        self.exp_id = 'test'
        self.core = {
            'network':{ 'ExpMode': 0, 'input_size': 2, 'output_size': 4, 'frame_num': 5,
                        'hidden_size': 5, 'output_dim': 8,
                        'NoiseStd': [0, 0, 0], 'Scale':[1, 2, 50], 
                        'Pro_kw':{ 'SigK': None}
                        },
            'data':{    'BaseFile': None,
                        'PropagationFile': "**.pth.tar",
                        'load_checkpoint_dir': "**.pth.tar",
                        'SOAFile':  "**.mat",
                        'CoeffFile': "**.mat"
            }
        }
        self.loss = {
            'regu': 1000, 'thers':  0.17, 'sp': 0, 'regu2': 40, 'regu3':5
        }
        self.lr = [1e-2, 1e-4]
        self.batch_size = 33
        super().__init__()