import torch
import logging
from tensorboardX import SummaryWriter
import random
import numpy as np
from tqdm import tqdm
import os
import json
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader
from network import RNN
import sys

def eval(output, label):
    top_n, top_i = output.topk(1)
    return (top_i[:, 0] == label).float().mean()

class BaseAgent:
    logger = logging.getLogger("Agent")
    current_epoch = 0
    current_iteration = 0
    Data_T = None
    Data_V = None
    scheduler = None
    criterian = nn.CrossEntropyLoss()
    relu = nn.ReLU(inplace=True)
    def __init__(self, params):       
        self.params = params
        self.manual_seed = self.params.manual_seed
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        self.device = params.device
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.network = None
        self.lr = self.params.lr
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir, comment='biscale')
        self.evaluate = eval
        

    def load_checkpoint(self, filename):
        if (filename is None):
            print('do not load checkpoint')
        else:
            try:
                print("Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                rnn_state = checkpoint['rnn']
                self.network.load_state_dict(rnn_state)
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.lr = checkpoint['lr']
                self.network.load_state_dict(checkpoint['rnn'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                      .format(self.params.core['data']['load_checkpoint_dir'], checkpoint['epoch'], checkpoint['iteration']))

            except OSError as e:
                self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
                self.logger.info("**First time to train**")


    def run(self):
        self.train()


    def train(self):
        os.makedirs(os.path.join(self.params.checkpoint_dir, ''), exist_ok=True)
        with open(os.path.join(self.params.checkpoint_dir, 'params.pth'), 'wb') as fp:
            pickle.dump(self.params, fp)
        with open(os.path.join(self.params.checkpoint_dir, 'core.json'), 'w') as fp:
            json.dump(self.params.core, fp)
        print('start training')
        DataL_T = DataLoader(self.Data_T, batch_size=self.params.batch_size, shuffle=True, num_workers=0)
        DataL_V = DataLoader(self.Data_V, batch_size=self.params.batch_size, shuffle=False, num_workers=0)
        self.network.train()
        for epoch in range(self.current_epoch, self.params.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch(DataL_T)
            if (epoch % self.params.save_checkpoint_iter == 0):
                self.save_checkpoint()
            if (epoch  % self.params.validate_iter == 0):
                torch.cuda.empty_cache()
                self.validate(DataL_V)
            if self.scheduler is not None:
                self.scheduler.step()
         
                      
    def validate(self, Dataset_test):
        log = AverageMeterList(2)
        self.network.eval()
        iteration = tqdm(Dataset_test, leave=False, desc='Val', ncols=120)
        with torch.no_grad():
            for index_dataset, samples in enumerate(iteration):
                torch.cuda.empty_cache()
                sample, label = samples
                sample = sample.to(self.device)
                label = label.to(self.device)
                output = self.network(sample)
                loss = self.criterian(output, label)
                accuracy = self.evaluate(output, label)
                log.update([loss.cpu().detach().numpy(),  
                            accuracy.cpu().detach().numpy()])
                iteration.set_postfix_str(f'{log}')
        loss_avg, acc_avg = log.avg()
        self.summary_writer.add_scalar("epoch/loss_validate", loss_avg, self.current_epoch)
        self.summary_writer.add_scalar("epoch/accuracy_validate", acc_avg, self.current_epoch)
        tqdm.write(f'Val: {log}')


    def train_one_epoch(self, Dataset):
        log = AverageMeterList(2)
        self.network.train()
        iteration = tqdm(Dataset, leave=False, desc=f'Train {self.current_epoch} epoch', ncols=120)
        for _, samples in enumerate(iteration):
            torch.cuda.empty_cache()
            sample, label = samples
            sample = sample.to(self.device) 
            label = label.to(self.device)
            self.optimizer.zero_grad()
            self.network.zero_grad()
            output = self.network(sample)
            loss = self.criterian(output, label)
            loss.backward()
            self.optimizer.step()
            accuracy = self.evaluate(output, label)
            log.update([loss.cpu().detach().numpy(),  
                        accuracy.cpu().detach().numpy()])
            self.current_iteration += 1
            iteration.set_postfix_str(f'{log}')
        loss_avg, acc_avg = log.avg()
        self.summary_writer.add_scalar("epoch/loss", loss_avg, self.current_epoch)
        self.summary_writer.add_scalar("epoch/accuracy", acc_avg, self.current_epoch)
        tqdm.write(f'Train {self.current_epoch} epoch: {log}')


    def update_settings(self):
        print(f'total params: {sum(param.numel() for param in self.network.parameters()):,}')
        self.optimizer = torch.optim.Adam([{
            "params": filter(lambda p: p.requires_grad, self.network.parameters()),
            "lr": self.params.lr
            # "weight_decay": 1e-5
        }])
        if self.params.core['data']['BaseFile'] is not None:
            filename = self.params.core['data']['BaseFile']
            checkpoint = torch.load(filename)
            state_dict = checkpoint['rnn']
            self.network.load_state_dict(state_dict)
        self.network.to(self.device)
        self.load_checkpoint(filename=self.params.core['data']['load_checkpoint_dir'])


    def save_checkpoint(self):
        file_name = 'epoch_%s.pth.tar' % str(self.current_epoch).zfill(5)

        state = {
            'epoch': self.current_epoch,
            'lr': self.lr,
            'iteration': self.current_iteration,
            'rnn': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        torch.save(state, os.path.join(self.params.checkpoint_dir, file_name))


class mylog:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.max = None
        self.min = None
    
    def __call__(self, x_max, x_min):
        self.max = torch.max(x_max, self.max) if self.max is not None else x_max
        self.min = torch.min(x_min, self.min) if self.min is not None else x_min
        return
    
    def __repr__(self):
        return f'{self.max:.4f}, {self.min:.4f}' 


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.4f}({:.2f}) '.format(self.avg, self.val)


class AverageMeterList(object):
    def __init__(self, len_):
        self.len_ = len_
        self.AML = [AverageMeter() for _ in range(len_)]
        self.reset()

    def reset(self):
        for AM in self.AML:
            AM.reset()

    def update(self, val_list, n=1):
        for val, AM in zip(val_list, self.AML):
            AM.update(val, n)

    def avg(self):
        return [AM.avg for AM in self.AML]

    def __repr__(self):
        res = ""
        for AM in self.AML:
            res += AM.__repr__()
        return res