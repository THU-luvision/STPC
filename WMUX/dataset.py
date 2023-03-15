import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import os
import hashlib
from PIL import Image
# from torchvision.datasets._optical_flow import FlowDataset
from os.path import join
import torchvision
import torch.nn.functional as F
import scipy.io as scio

class SimpleData(Dataset):
    def __init__(self, split='train', shuffle=False):
        super().__init__()
        data_raw = np.array([ [2, 1, 0],
                              [2, 0, 1],
                              [0, 2, 1],
                              [1, 0, 2],
                              [1, 2, 0],
                              [0, 1, 2],
                              ])
        label_raw = np.array([0, 0, 0, 1, 1, 1])
        data, label = [], []
        if split != 'test':
            self.N = 30 if split=='train' else 10
            np.random.seed(self.N)
            for i in range(label_raw.shape[0]):
                label += [label_raw[i]] * self.N
                tmp = np.zeros((self.N, 3))
                tmp[:, data_raw[i]!=0] = np.random.rand(self.N, 2) * 0.1 + 0.9
                data.append(np.stack([np.repeat(data_raw[i].reshape((1, -1)), self.N, axis=0), tmp], axis=1))
            self.label = np.array(label).ravel()
            self.data = np.concatenate(data, axis=0).reshape((-1, 2, 3)) # b, 2, 3
        else:
            self.N = 1
            np.random.seed(self.N)
            for i in range(9):
                label.append(0)
                tmp = np.zeros((1, 3))
                tmp[:, data_raw[i % 3]!=0] = np.random.rand(self.N, 2) * 0.1 + 0.9
                data.append(np.stack([data_raw[i % 3].reshape((1, -1)), tmp], axis=1))
            label.append(1)
            tmp = np.zeros((1, 3))
            tmp[:, data_raw[4]!=0] = np.random.rand(self.N, 2) * 0.1 + 0.9
            data.append(np.stack([data_raw[4].reshape((1, -1)), tmp], axis=1))
            
            self.label = np.array(label).ravel()
            self.data = np.concatenate(data, axis=0).reshape((-1, 2, 3)) # b, 2, 3
            
        if shuffle:
            np.random.seed(20)
            index = np.arange(self.label.shape[0], dtype=np.int_)
            np.random.shuffle(index)
            self.label = self.label[index]
            self.data = self.data[index]
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (torch.tensor(self.data[index, ...], dtype=torch.float32), 
                torch.tensor(self.label[index], dtype=torch.long))
        
class ExpData(Dataset):
    def __init__(self, datadir):
        d = scio.loadmat(datadir + 'sys_res.mat')
        data = np.stack([d['y1_i'].ravel(), d['y2_i'].ravel()], axis=1) # b, 2
        sw = np.array([d['y1_o'][0, 0], d['y2_o'][0, 0]])
        self.data = np.stack([data, np.repeat(sw[None, :], data.shape[0], axis=0)], axis=1) # b, 2, 2
        self.label = np.stack([d['y1_er'].ravel(), d['y2_er'].ravel()], axis=1) # b, 2
        
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float32), 
                torch.tensor(self.label[index], dtype=torch.float32))

if __name__ == '__main__':
    Data_T = SimpleData(split='train')
    Data_V = SimpleData(split='val')
    scio.savemat('fulldata.mat', {'TrainData': Data_T.data, 'TrainLabel': Data_T.label,
                                  'ValData': Data_V.data, 'ValLabel': Data_V.label})