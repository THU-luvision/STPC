from os import stat
from typing import OrderedDict
from numpy.lib.type_check import real
from torch._C import device
from torch.functional import align_tensors
import torch.nn as nn
import torch
from scipy import io
import numpy as np
import collections
from math import pi
import pickle
import scipy.io as scio

class Propagation130(nn.Module):
    def __init__(self, params, ExpMode=0, NoiseStd=0, Scale=1., SigK=1):
        super(Propagation130, self).__init__()
        random_mask = torch.randn(params.core['network']['output_dim'], params.core['network']['frame_num'], params.core['network']['input_size'], 
                                  26, 26, requires_grad=True).to(params.device)  + 1
        random_mask = random_mask * torch.sgn(torch.randn_like(random_mask)).to(params.device)  
        self.random_mask = nn.Parameter(random_mask)
        self.device = params.device
        filename = params.core['data']['PropagationFile']
        checkpoint = torch.load(filename)
        self.Propagation_w = checkpoint['rnn']['w'].reshape((1, 1, 1, 1, -1))
        self.Propagation_b = checkpoint['rnn']['b'].reshape((1, 1, 1, 1, -1))
        self.sigmoid = nn.Sigmoid()
        self.ExpMode = ExpMode
        self.NoiseStd = NoiseStd
        self.SigK = SigK
        self.Scale = Scale

    def forward(self, x):        
        # x (b, f, img_size[0], img_size[1]) --> x(b, f, rm_size[0], rm_size[1])
        # random_mask (f, input_size, rm_size[0], rm_size[1])
        # x*rm (b, f, input_size, rm_size[0], rm_size[1]) --> (b, f*input_size, rm_size[0], rm_size[1])
        if self.ExpMode < 2:
            if self.ExpMode < 1 and self.SigK is not None:
                rm = self.sigmoid(self.random_mask * self.SigK)
            else:
                rm = (torch.sgn(self.random_mask) + 1) / 2
            x = 1. - x
            # (b, f, input_size, rm_size[0], rm_size[1])
            x = torch.nn.functional.interpolate(x, size=rm.shape[-2:], mode='nearest')
            x = x.unsqueeze(2).unsqueeze(1) * rm.unsqueeze(0) # (b, od, f, m, 26, 26)
            x = x.flatten(2, 3) # (b, od, fm, 26, 26)
        
        rm_x = x.flatten(3, 4).type(torch.complex128).unsqueeze(-1) # (b, od, fm, h*w, 1)
        rm_x = (torch.matmul(self.Propagation_w, rm_x) + self.Propagation_b).squeeze(-1) # (b, od, fm, 1)
        rm_x = torch.square(torch.abs(rm_x)).float().squeeze(-1)# (b, od, fm)
        rm_x = rm_x + torch.randn_like(rm_x).to(self.device) * self.NoiseStd
        rm_x = rm_x * self.Scale
        
        if self.ExpMode < 1:
            return rm_x
        elif self.ExpMode == 1:
            return rm_x, x
        elif self.ExpMode == 2:
            return rm_x, rm_x


class SOAmultiply(nn.Module):
    def __init__(self, params, input_size, output_size, NoiseStd=0, Scale=10):
        super(SOAmultiply, self).__init__()
        filename = params.core['data']['SOAFile']
        if filename[-4:] == '.mat':
            data = scio.loadmat(filename)
        else:
            data = torch.load(filename)
        in_ = data['in_'].flatten()[::-1].copy()
        self.x_ = torch.from_numpy(in_).to(params.device)
        self.x_2 = torch.from_numpy(np.append(in_, 1)).to(params.device)
        self.x_step = (self.x_.shape[0] - 1) / 2
        self.y_ = data['weight'].flatten()
        self.y_mean = np.mean(self.y_)
        self.y_range = np.max(self.y_) - np.min(self.y_)
        self.z_ = torch.from_numpy(np.repeat(data['out_'][None, None, ...], input_size, axis=0)).to(params.device)
        self.NoiseStd = NoiseStd
        self.device = params.device
        self.output_size = output_size
        self.batch_num = params.batch_size
        self.Scale = np.mean(scio.loadmat(params.core['data']['CoeffFile'])['coeff'][2, :])
        
    def forward(self, weight, x):
        """
        Args:
            weight : input, output
            x : batch, input
            
        Returns: input, batch, output
        """
        y_index = torch.abs(weight.unsqueeze(1))
        y_index = 2 * (self.y_mean - y_index) / self.y_range # input, 1, output
        
        x = x.permute(1, 0).unsqueeze(2) # input, batch, 1
        x_arg = torch.argmin(torch.abs(x - self.x_[None, None, ...]), dim = 2, keepdim=True)
        x_arg[self.x_[x_arg] > x] = x_arg[self.x_[x_arg] > x] - 1
        x_index = (x - self.x_2[x_arg]) / (self.x_2[x_arg + 1] - self.x_2[x_arg]) / self.x_step + x_arg / self.x_step - 1# input, batch, 1 
        x_index[x > self.x_[-1]] = 1
        x_index[x < self.x_[0]] = -1
        x_index = - x_index
        index = torch.stack([x_index.repeat(1, 1, y_index.shape[2]), 
                             y_index.repeat(1, x_index.shape[1], 1)], dim = -1) # input, batch, output, 2
        res = nn.functional.grid_sample(self.z_, index, align_corners=True, padding_mode='border').squeeze(1) #input, batch,  output
        res = res + torch.randn_like(res).to(self.device) * self.NoiseStd
        return res * self.Scale


class SinSquare:
    def __init__(self, params, NoiseStd=0):
        super(SinSquare, self).__init__()
        coeff = scio.loadmat(params.core['data']['CoeffFile'])['coeff'][:2, :]
        if coeff.shape[-1] < 5:
            coeff = np.repeat(coeff, 5 // coeff.shape[-1] + 1, 1)
        self.coeff = torch.tensor(coeff[:, :5], device=params.device).reshape(2, 1, 1, 5)
        self.device = params.device
        self.NoiseStd = NoiseStd
    
    def __call__(self, x):
        '''
        x: batch, output
        coeff[0], coeff[1]: 1, output
        '''
        coeff = self.coeff
        x_in = torch.abs(x)
        y = coeff[0] * torch.square(torch.sin(x_in * coeff[1]))
        y = y * torch.sgn(x) + torch.randn_like(y).to(self.device) * self.NoiseStd
        return y
    

class LOOPaddition:
    def __init__(self, params, ExpMode, **kwargs):
        self.device = params.device
        self.SFun = SinSquare(params, **kwargs)
        self.ExpMode = ExpMode
    
    def __call__(self, x): # x (input, batch, outdim, output)
        if self.ExpMode == 5:
            y = x
        else:
            y_log = []
            y = torch.zeros(x.shape[1:], device=self.device)
            for i in range(x.shape[0]):
                in_ = y + x[i]
                y = self.SFun(in_)
                if self.ExpMode == 0:
                    y_log.append(in_.clone())
                elif self.ExpMode == 4:
                    if i != x.shape[0] - 1:# just for show
                        y_log.append(y.clone() + x[i+1, ...] * 14.3 / 58)
                    else:
                        y_log.append(y.clone()) 
        if self.ExpMode == 4 or self.ExpMode == 0:
            y_log = torch.cat(y_log, dim=-1)
            return y, y_log # batch, 2, 100
        else:
            return y


class RealLinear(nn.Module):
    def __init__(self, params, input_size, output_size, output_dim, ExpMode=0, NoiseStd=[0, 0], Scale=[20, 50], **kwargs):
        super(RealLinear, self).__init__()
        self.ExpMode = ExpMode
        self.weight = nn.Parameter(torch.randn(output_dim, input_size, output_size, device=params.device))
        self.Multiply = SOAmultiply(params, input_size*output_dim, output_size, NoiseStd[0], Scale[0])
        self.Addition = LOOPaddition(params, ExpMode=ExpMode, NoiseStd=NoiseStd[1], **kwargs) # scale, boundary
        self.Scale = Scale[1]
         
    
    def forward(self, x): # x: batch, od, fm, w: od, fm, os
        
        if self.ExpMode < 4:
            weight = torch.abs(self.weight)
            res_ = self.Multiply(weight.flatten(0, 1), x.flatten(1, 2)) # od*fm, batch, os
            res_ = res_.reshape(x.shape[1], x.shape[2], -1, weight.shape[-1]).permute(1, 2, 0, 3) # fm, batch, od, os
        elif self.ExpMode == 4:
            res_ = x.permute(2, 0, 1, 3) # input, batch, outd, output
            
        if self.ExpMode == 4 or self.ExpMode == 0:
            res, y_log = self.Addition(res_)
        elif self.ExpMode == 5:
            res = x
        else:
            res = self.Addition(res_)
            
        res = res * self.Scale
        
        if self.ExpMode == 0:
            return res, y_log
        elif self.ExpMode < 3:
            return res
        elif self.ExpMode == 3:
            res_ = res_.permute(1, 2, 0, 3)# batch, outd, input, output
            return res, res_
        elif self.ExpMode == 4:
            return res, y_log # batch, 2, output*10
        elif self.ExpMode == 5:
            return res, res


class RNN_reallight(nn.Module):
    def __init__(self, params, ExpMode=0, 
                 input_size=10, hidden_size=5, output_size=10, frame_num=1, output_dim=2,
                 NoiseStd=[0, 0, 0], Scale=[1.2, 20, 50], Pro_kw=None):
        # random_mask: (frame_num, h, w, input_size)
        super(RNN_reallight, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.ExpMode = ExpMode
        self.device = params.device
        self.Propagation = Propagation130(params, ExpMode, NoiseStd[0], Scale[0], **Pro_kw)
        self.i2i = RealLinear(params, frame_num * input_size, hidden_size, output_dim, ExpMode, NoiseStd[1:], Scale[1:])
        self.i2o = nn.Linear(output_dim * hidden_size, output_size, dtype=torch.double)
        torch.nn.init.normal_(self.i2i.weight, mean=self.i2i.Multiply.y_mean, std=self.i2i.Multiply.y_range / 2  / 3)
        torch.nn.init.xavier_normal_(self.i2o.weight)

  

    def Change(self, ExpMode=None, batch_num=None,
               NoiseStd=None, Scale=None, SigK=None, Boundary=None,
               CoeffFile=None, SOAFile=None, PropagationFile=None):
        if ExpMode is not None:
            self.ExpMode = ExpMode 
            self.Propagation.ExpMode = ExpMode
            self.i2i.ExpMode = ExpMode
            self.i2i.Addition.ExpMode = ExpMode
            
        if NoiseStd is not None:
            self.Propagation.Noise_std = NoiseStd[0] if NoiseStd[0] is not None else self.Propagation.Noise_std
            self.i2i.Multiply.NoiseStd = NoiseStd[1] if NoiseStd[1] is not None else self.i2i.Multiply.NoiseStd
            self.i2i.Addition.SFun.NoiseStd = NoiseStd[2] if NoiseStd[2] is not None else self.i2i.Addition.SFun.NoiseStd
            
        self.i2i.Multiply.batch_num = batch_num if batch_num is not None else self.i2i.Multiply.batch_num
        
        if Scale is not None:
            self.Propagation.Scale = Scale[0] if Scale[0] is not None else self.Propagation.Scale
            self.i2i.Multiply.Scale = Scale[1] if Scale[1] is not None else self.i2i.Multiply.Scale
            self.i2i.Scale = Scale[2] if Scale[2] is not None else self.i2i.Scale
            
        self.Propagation.SigK = SigK if SigK is not None else self.Propagation.SigK
        if CoeffFile is not None:
            rcoeff = scio.loadmat(CoeffFile)['coeff']
            self.i2i.Addition.SFun.coeff =  torch.tensor(rcoeff[:2, :], device=self.device).reshape(2, 1, 1, 5)
            self.i2i.Multiply.Scale = np.mean(rcoeff[2, :])
            
        if SOAFile is not None:
            if SOAFile[-4:] == '.mat':
                data = scio.loadmat(SOAFile)
            else:
                data = torch.load(SOAFile)
            in_ = data['in_'].flatten()[::-1].copy()
            self.i2i.Multiply.x_ = torch.from_numpy(in_).to(self.device)
            self.i2i.Multiply.x_2 = torch.from_numpy(np.append(in_, 1)).to(self.device)
            self.i2i.Multiply.x_step = (self.i2i.Multiply.x_.shape[0] - 1) / 2
            self.i2i.Multiply.y_ = data['weight'].flatten()
            self.i2i.Multiply.y_mean = np.mean(self.i2i.Multiply.y_)
            self.i2i.Multiply.y_range = np.max(self.i2i.Multiply.y_) - np.min(self.i2i.Multiply.y_)
            self.i2i.Multiply.z_ = torch.from_numpy(np.repeat(data['out_'][None, None, ...], self.i2i.Multiply.z_.shape[0], axis=0)).to(self.device)
        
        if PropagationFile is not None:
            checkpoint = torch.load(PropagationFile)
            self.Propagation.Propagation_w = checkpoint['rnn']['w'].flatten()[None, None, None, ...]
            self.Propagation.Propagation_b = checkpoint['rnn']['b'].flatten()[None, None, None, ...]
        return

    def forward(self, x):
        """
        x: (batch , frame_num, 26, 26)
        """
        if self.ExpMode == 0:
            x = self.Propagation(x)
            output, y_log = self.i2i(x)
        elif self.ExpMode < 1:
            x = self.Propagation(x)
            output = self.i2i(x)
        elif self.ExpMode < 3:
            x, pro_out = self.Propagation(x)
            output = self.i2i(x)
        else:
            output, soa_output = self.i2i(x)
        
        output_ = self.i2o(output.flatten(1, 2))
        
        if self.ExpMode == 0:
            return output_, x, y_log
        elif 1 <= self.ExpMode <= 2:
            return output_, pro_out
        elif 3 <= self.ExpMode <= 5:
            return output_, soa_output
        elif self.ExpMode == -1:
            return output_, output
    
