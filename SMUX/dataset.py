import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
import os
import hashlib
from PIL import Image
from os.path import join
import torchvision
import torch.nn.functional as F
import scipy.io as scio

class MOV_DataSet(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.split = split
        with open('data16.pkl', 'rb') as fp:
            All = pickle.load(fp)
        if split == 'train':
            self.label = All['TrainLabel']
            self.sample = All['TrainSample'] # length, 120, 160, 5
        elif split == 'val':
            self.label = All['ValLabel']
            self.sample = All['ValSample']
        self.length = self.label.shape[0]
        self.rm_size = [26, 26]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img = self.sample[index, ...]
        label = self.label[index]
        print(index, end=', ')
        return (torch.from_numpy(img).float(), torch.tensor(label).long())
  

if __name__ == '__main__':
    pass