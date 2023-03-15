from dataset import *
from parameter import *
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tqdm import tqdm
from utils import *
import scipy.io as scio
from network2 import *

class TowardsAgent(BaseAgent):
    def __init__(self, params):
        super().__init__(params)
        self.Data_T = SimpleData(split='train')
        self.Data_V = SimpleData(split='val')
        self.criterian = Myloss()
        self.evaluate = Myeval
        self.network = RNN(params)
        self.update_settings()


    
class EXPagent(BaseAgent):
    def __init__(self, params):
        super().__init__(params)      
        self.Data_V = SimpleData(split='val')
        self.Data_T = SimpleData(split='train')
        self.Data_Te = SimpleData(split='test', shuffle=True)
        self.network = ExpRNN(params)
        self.update_settings()
        
    def _generate_exp_data_1(self, bs, sys=False, exp_num=1, split='val'):
        if sys:
            DataL_V = DataLoader(self.Data_T, batch_size=20, shuffle=False, num_workers=0)
        else:
            if split == 'val':
                DataL_V = DataLoader(self.Data_V, batch_size=len(self.Data_V), shuffle=False, num_workers=0)
            elif split == 'test':
                DataL_V = DataLoader(self.Data_Te, batch_size=len(self.Data_Te), shuffle=False, num_workers=0)
        log = AverageMeterList(2)
        self.network.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            sample, label = DataL_V.__iter__().__next__()
            self.label = label.clone()
            sample = sample.to(self.device)
            label = label.to(self.device)
            output, exp_data = self.network(sample, exp_num=-1)
            loss = self.criterian(output, label)
            accuracy = self.evaluate(output, label)
            log.update([loss.cpu().detach().numpy(), accuracy.cpu().detach().numpy()])
        tqdm.write(f'Val: {log}')
        
        y1, y2, tw = exp_data
        y1_i, y1_e, y1_r, y10, y1s, y1_o = y1 # b, 3
        y2_i, y2_e, y2_r, y20, y2s, y2_o = y2
        if sys:
            y1_r = y1_r.reshape((-1, bs))
            y2_r = y2_r.reshape((-1, bs))
            y1_e[:, 0] = y1_e[y1_e != 0]
            y1_e[:, 1:] = 0
            y2_e[:, 0] = y2_e[y2_e != 0]
            y2_e[:, 1:] = 0
            y1_e = y1_e.reshape((-1, bs, 3, 1))
            y2_e = y2_e.reshape((-1, bs, 3, 1))
            y1_i = y1_i[y1_i!=0]
            y2_i = y2_i[y2_i!=0]
            return (y1_i, y1_e, y1_r, y1_o), (y2_i, y2_e, y2_r, y2_o)
        else:
            y1_e = y1_e.reshape((-1, bs, 3, 1))
            y2_e = y2_e.reshape((-1, bs, 3, 1))
            y10 = y10.reshape((-1, bs, 2))
            y20 = y20.reshape((-1, bs, 2))
            y1s = y1s.reshape((-1, bs, 2))
            y2s = y2s.reshape((-1, bs, 2))
            tw = np.repeat(tw.reshape((1, 3, 2)), repeats=bs, axis=0)
            gt = (y10, y20) if exp_num == 1 else (y1s, y2s)
            gt = np.stack(gt, axis=0) # 2, 6, 10, 2
            return (y1_e, y1_o), (y2_e, y2_o), tw, gt

    def _calculate_exp_res_1(self, exp_res, exp_num=1):
        exp_res = torch.tensor(exp_res.reshape((2, -1, 2)), dtype=torch.float32, device=self.device)
        log = AverageMeterList(2)
        self.network.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            # _, label = self.DataL_V.__iter__().__next__()
            label = self.label.to(self.device)
            output = self.network(exp_res, exp_num=exp_num)
            loss = self.criterian(output, label)
            accuracy = self.evaluate(output, label)
            log.update([loss.cpu().detach().numpy(), accuracy.cpu().detach().numpy()])
        tqdm.write(f'Val: {log}')
        return
    
    def _generate_exp_data_2(self, savedir, split='val'):
        d = scio.loadmat(f'./data/{savedir}/{split}/1_2_net_res.mat')
        self.label = torch.tensor(d['label']).ravel()
        y0 = d['res'] / 10 
        y0 = np.stack([y0[0, ..., 0], y0[1, ..., 1]], axis=2) # 6, 10, 2
        
        tw = self.network.t_weight2
        tw = torch.sigmoid(tw).cpu().detach().numpy()
        
        gt = np.matmul(y0, tw) # 6, 10, 2
        tw = self.exp_i2v_tw(tw)
        tw = np.concatenate([tw, np.zeros((1, 2))], axis=0)[None, ...] # 1, 3, 2
        tw = np.repeat(tw, gt.shape[-2], axis=0) # 10, 3, 2
        
        y0_v = self.exp_i2v(y0) # 2, 6, 10, 3, 1
        return y0_v, gt, tw
        
    def _calculate_exp_res_2(self, exp_res):
        exp_res = exp_res.reshape((-1, 2))
        log = AverageMeterList(2)
        label = self.label.to(self.device)
        output = torch.tensor(exp_res, dtype=torch.float32, device=self.device)
        loss = self.criterian(output, label)
        accuracy = self.evaluate(output, label)
        log.update([loss.cpu().detach().numpy(), accuracy.cpu().detach().numpy()])
        tqdm.write(f'Val: {log}')

    
    def exp_i2v(self, exp_y0):
        # 6, 10, 2
        d1 = scio.loadmat(self.params.core['data']['DataDir'] + '1_res_m.mat')
        d2 = scio.loadmat(self.params.core['data']['DataDir'] + '2_res_m.mat')
        d1['v2i'][:, 1] = d1['v2i'][:, 1] * d1['i_max']
        d2['v2i'][:, 1] = d2['v2i'][:, 1] * d2['i_max']
        v2i = [d1['v2i'], d2['v2i']]
        alpha = scio.loadmat(self.params.core['data']['DataDir'] + '1_res_m.mat')['alpha'].ravel()
        
        y0_v = np.zeros((2, *exp_y0.shape[:2], 3, 1)) # 2, 10, 2, 3, 1
        exp_y0[..., 0] = exp_y0[..., 0] / alpha[1]
        for j in range(2):
            assert(np.max(exp_y0[..., j].ravel()) <= v2i[j][0, 1])
            y0_v[j, ..., j, 0] = np.interp(exp_y0[..., j], v2i[j][:, 1][::-1], v2i[j][:, 0][::-1])
        return y0_v

    def exp_i2v_tw(self, tw):
        res = scio.loadmat(self.params.core['data']['DataDir'] + '3_res.mat')['res']
        res[:, 1] = res[:, 1] / res[0, 1]
        return np.interp(tw, res[:, 1][::-1], res[:, 0][::-1])


class Myloss:
    loss = nn.CrossEntropyLoss()
    def __call__(self, output, label):
        if isinstance(output, tuple):
            output, regu = output
            return self.loss(output, label) + regu 
        else:
            return self.loss(output, label)
        

def Myeval(output, label):
    if isinstance(output, tuple):
        output, regu = output
    top_n, top_i = output.topk(1)
    return (top_i[:, 0] == label).float().mean()

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    params = ExpParams()
    agent = TowardsAgent(params)
    agent.train()
