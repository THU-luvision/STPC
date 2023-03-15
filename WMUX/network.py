from re import L
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.io as scio

def T2N(x):
    return x.cpu().detach().numpy()

class grid_sample(nn.Module):
    def __init__(self, datadir, device):
        super().__init__()
        d = scio.loadmat(datadir)
        self.x1 = torch.tensor(d['o'][:, 0, 0].flatten(), device=device, dtype=torch.float32)
        self.x2 = torch.tensor(d['o'][0, :, 1].flatten(), device=device, dtype=torch.float32)
        self.y = torch.tensor(d['res'], device=device, dtype=torch.float32)
        if 'alpha' in d.keys():
            self.alpha = torch.tensor(d['alpha'].ravel(), device=device, dtype=torch.float32)[None, :]
            self.alpha = torch.flip(self.alpha, [1])
        
    def forward(self, x1, x2):
        regu = torch.mean((torch.relu(x1 - self.x1[0] - 0.001) + torch.relu(self.x1[-1] + 0.001 - x1))) / torch.mean(x1) / 2 + \
               torch.mean((torch.relu(x2 - self.x2[0] - 0.001) + torch.relu(self.x2[-1] + 0.001 - x2))) / torch.mean(x2) / 2
        # batch, 3
        x1_index = self._generate_index(x1, self.x1) # b
        x2_index = self._generate_index(x2, self.x2) # b
        index = torch.stack([x2_index, x1_index], dim=1)[None, None, ...] # 1, 1, b, 2
        y = nn.functional.grid_sample(self.y[None, None, ...], 
                                      index, align_corners=True, padding_mode='border')[0, 0, 0]
        return y, regu
    
    def _generate_index(self, x, x0):
        # x(b), x0(20)
        step = (x0.shape[0] - 1.) / 2
        x = x.clamp(x0[-1], x0[0])
        arg = torch.argmin(torch.abs(x[..., None] - x0[None, ...]), dim=1, keepdim=False)
        arg[x0[arg] > x] += 1
        index = (x0[arg-1] - x) / (x0[arg - 1] - x0[arg]) / step + (arg - 1) / step - 1
        return index


class RNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.s_weight = nn.Parameter(torch.randn(2, device=params.device, dtype=torch.float32))
        self.t_weight = nn.Parameter(torch.randn(3, 2, device=params.device, dtype=torch.float32))
        self.t_weight2 = nn.Parameter(torch.randn(2, 2, device=params.device, dtype=torch.float32))
        self.device=params.device
        
        self.gs_sw1 = grid_sample(params.core['data']['1_res_dir'], params.device)
        self.gs_sw2 = grid_sample(params.core['data']['2_res_dir'], params.device)
        self.gs_soa1 = grid_sample(params.core['data']['DataDir'] + '1_soares.mat', params.device)
        self.gs_soa2 = grid_sample(params.core['data']['DataDir'] + '2_soares.mat', params.device)
        
    def forward(
        self, 
        x
        ):
        # x: b, 2, 3
        sw = torch.sigmoid(self.s_weight)
        tw = torch.sigmoid(self.t_weight)
        flag1, flag2 = x[:, 0] == 1, x[:, 0] == 2
        
        y1 = flag1.clone().type(torch.float32)
        y1[flag1] = x[:, 1][flag1]
        y1[flag1], _ = self.gs_sw1(y1[flag1], sw[0].expand(y1.shape[0]))
        y1 = y1 * self.gs_sw1.alpha
        y1 = torch.matmul(y1, tw)
        
        y2 = flag2.clone().type(torch.float32)
        y2[flag2] = x[:, 1][flag2]
        y2[flag2], _ = self.gs_sw2(y2[flag2], sw[1].expand(y2.shape[0]))
        y2 = y2 * self.gs_sw2.alpha
        y2 = torch.matmul(y2, tw)
        
        y_1, regu1 = self.gs_soa1(y1[:, 0], y2[:, 0])
        y_2, regu2 = self.gs_soa2(y1[:, 1], y2[:, 1])
        regu = regu1 + regu2
        y = torch.stack([y_1, y_2], dim=-1)
        tw2 = torch.sigmoid(self.t_weight2)
        
        y = torch.matmul(y, tw2)
        return y, regu
    

class ExpRNN(RNN):
    def __init__(self, params):
        super().__init__(params)
        self.v2i = []
        for i in range(1, 4):
            if i < 3:
                tmp = scio.loadmat(params.core['data']['DataDir'] + f'{i}_res_m.mat')['v2i']
            else:
                tmp = scio.loadmat(params.core['data']['DataDir'] + f'{i}_res.mat')['res']
            if i == 3:
                tmp[:, 1] = tmp[:, 1] / tmp[0, 1]
            self.v2i.append(tmp)
        # 20, 2; 20, 2; 20, 2
    
    def forward(self, x, exp_num=0):
        # x: b, 3
        if  exp_num == 2:
            y1s, y2s = x[0, ...], x[1, ...]
        else:
            if exp_num == 1:
                y1, y2 = x[0, ...], x[1, ...]
            else:
                sw = torch.sigmoid(self.s_weight)
                tw = torch.sigmoid(self.t_weight)
                flag1, flag2 = x[:, 0] == 1, x[:, 0] == 2
                
                y1 = flag1.clone().type(torch.float32)
                y1[flag1] = x[:, 1][flag1]
                y10 = y1.clone()
                y1[flag1], _ = self.gs_sw1(y1[flag1], sw[0].expand(y1.shape[0]))
                y10_r = y1[flag1].clone()
                y1 = y1 * self.gs_sw1.alpha 
                y1 = torch.matmul(y1, tw)
                
                y2 = flag2.clone().type(torch.float32)
                y2[flag2] = x[:, 1][flag2]
                y20 = y2.clone()
                y2[flag2], _ = self.gs_sw2(y2[flag2], sw[1].expand(y2.shape[0]))
                y20_r = y2[flag2].clone()
                y2 = y2 * self.gs_sw2.alpha
                y2 = torch.matmul(y2, tw)
        
            y1s, _ = self.gs_soa1(torch.ravel(y1), torch.ravel(y2))
            y2s, _ = self.gs_soa2(torch.ravel(y1), torch.ravel(y2))
            y1s = y1s.reshape((-1, 2))
            y2s = y2s.reshape((-1, 2))
            
        y0 = torch.stack([y1s[:, 0], y2s[:, 1]], dim=-1)
        tw2 = torch.sigmoid(self.t_weight2)
        y = torch.matmul(y0, tw2)
        
        if exp_num != -1:
            return y
        else:
            y10, y10_r, y1, y1s = T2N(y10), T2N(y10_r), T2N(y1), T2N(y1s)
            y20, y20_r, y2, y2s = T2N(y20), T2N(y20_r), T2N(y2), T2N(y2s)
            sw, tw = T2N(sw), T2N(tw)
            return y, (
                   (y10, self._i2v(y10, port=0), y10_r, y1, y1s, sw[0]),
                   (y20, self._i2v(y20, port=1), y20_r, y2, y2s, sw[1]), self._i2v(tw, port=2))

    def _i2v(self, i, port=0):
        # i np.array
        assert(np.mean(np.logical_and(i>=0, i<=self.v2i[port][0, 1])) == 1)
        r = np.interp(i, self.v2i[port][:, 1][::-1], self.v2i[port][:, 0][::-1])
        r[i==0] = 0
        return r

if __name__ == '__main__':
    pass