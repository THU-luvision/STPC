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
from network import *
from copy import deepcopy

class SimAgent(BaseAgent):
    def __init__(self, params):
        super().__init__(params)
        self.Data_T = MOV_DataSet(split='train')
        self.Data_V = MOV_DataSet(split='val')
        self.criterian = Myloss(**params.loss)
        self.evaluate = Myeval
        self.network = RNN_reallight(params, **params.core['network'])
        self.update_settings()
        
    def update_settings(self):
            print(f'total params: {sum(param.numel() for param in self.network.parameters()):,}')
            self.optimizer = torch.optim.Adam([{
                        "params": self.network.Propagation.random_mask,
                        "lr": self.params.lr[0],
                        "weight_decay": 0
                    },
                    {
                        "params": [param for name, param in self.network.named_parameters() if 'random_mask' not in name],
                        "lr": self.params.lr[1],
                        "weight_decay": 0 
                    }])
            if self.params.core['data']['BaseFile'] is not None:
                filename = self.params.core['data']['BaseFile']
                checkpoint = torch.load(filename)
                state_dict = checkpoint['rnn']
                self.network.load_state_dict(state_dict, strict=False)
            self.network.to(self.device)
            self.load_checkpoint(filename=self.params.core['data']['load_checkpoint_dir'])


class EXPagent(SimAgent):
    gw = generate_weight()
    def __init__(self, params):
        super().__init__(params)
        self.criterian = nn.CrossEntropyLoss(ignore_index=-1)
        filename = params.core['data']['load_checkpoint_dir']
        self.root_dir = f'data\\{params.exp_id}\\'
        checkpoint = torch.load(filename)
        torch.save(checkpoint, self.root_dir + "model\\model.pth.tar")
        with open(self.root_dir + "model\\params.pth", 'wb') as fp:
            pickle.dump(params, fp)
        with open(self.root_dir + "model\\core.json", 'w') as fp:
            json.dump(params.core, fp)
        with open(self.root_dir + "model\\info.txt", 'w') as fp:
            fp.write(filename)
        self.scale = float(self.network.i2i.Multiply.Scale)
        
    
    def generate_DMDimg(self, sys_i=2, test_i=7, refresh=False):
        if refresh or not os.path.exists(self.root_dir + f"0_img_alls.pt"):
            Dl_V = DataLoader(self.Data_V, batch_size=self.params.batch_size, shuffle=True, num_workers=0)
            Dl_T = DataLoader(self.Data_T, batch_size=self.params.batch_size, shuffle=True, num_workers=0)
            self.label_s, self.img_s, img_alls = self._expmode1_DMDin(sys_i, Dl_T, 'Train', 's', True)
            self.label_t, self.img_t, img_allt = self._expmode1_DMDin(test_i, Dl_V, 'Test', 't', True)
        else:
            self.label_s, self.img_s, img_alls = self._expmode1_DMDin(sys_i, None, 'Train', 's', False)
            self.label_t, self.img_t, img_allt = self._expmode1_DMDin(test_i, None, 'Test', 't', False)
        return img_alls, img_allt
            

    def generate_weight(self, refresh):
        if refresh or not os.path.exists(self.root_dir + f"0_weight.npy"):
            weight = self.network.i2i.weight.abs().clamp(0, 5).cpu().detach().numpy() # od, ins, outs
            weight = self.gw(weight) # od, 5000
            
            tmp = np.linspace(5, 0, 11)[:10] # 10
            weight_ref = np.zeros((10, 10, 5))
            # weight_ref[:, 0, :] = (tmp[:, None] / 5) ** 0.25 * 5
            weight_ref[:, 0, :] = tmp[:, None]
            weight_ref = self.gw(weight_ref) # 10, 5000
            
            np.save(self.root_dir + f"0_weight.npy", weight)
            np.save(self.root_dir + f"0_weight_ref.npy", weight_ref)
        else:
            weight = np.load(self.root_dir + f"0_weight.npy")
            weight_ref = np.load(self.root_dir + f"0_weight_ref.npy")
        return weight, weight_ref
    
    def modes_eval(self, noise_std=[0.002, 0.02, 0.2]):
        mode1_sim = self._expmode(2, self.img_s, True, savedir='model\\mode1_sim0.npy', print_flag=False)
        noise = np.random.randn(*mode1_sim.shape) * noise_std[0]
        self._plot_re(mode1_sim, mode1_sim + noise, self.root_dir + 'model\\mod1_t.png')
        self._expmode(3, mode1_sim + noise, True, True, 'mode1 sim')
        
        mode2_sim = self._expmode(3, mode1_sim, True, savedir='model\\mode2_sim0.npy', print_flag=False)
        noise = np.random.randn(*mode2_sim.shape) * noise_std[1]
        self._plot_re(mode2_sim, mode2_sim + noise, self.root_dir + 'model\\mode2_t.png')
        self._expmode(4, mode2_sim + noise, True, True, 'mode2 sim')
        
        mode4_sim = self._expmode(4, mode2_sim, True, savedir='model\\mode4_sim0.npy', print_flag=False)
        noise = np.random.randn(*mode4_sim.shape) * noise_std[2]
        self._plot_re(mode4_sim, mode4_sim + noise, self.root_dir + 'model\\mode4_t.png')
        self._expmode(5, (mode4_sim + noise)[..., -5:], True, True, 'mode4 sim')
        return
        
    
    def mode1_eval(self, mode1_res, data_dict, sys, c_scale=1.): # DMD out
        self._change_datafile(data_dict)
        mode1_sim = self._expmode(2, self.img_s, True, 
                                  savedir=f'mode1_sim{int(sys)}.npy')
        mode1_res = np.transpose(mode1_res, [0, 2, 1, 3]) * c_scale
        self._plot_re(mode1_res, mode1_sim, self.root_dir + f'mode1_re{int(sys)}.png')
        self._expmode(3, mode1_sim, True, True, 'mode1 sim')
        self._expmode(3, mode1_res, True, True, 'mode1 res')
        return
    
    
    def mode2_eval(self, mode2_res, data_dict, prev_flag=-1, sys=False, s_scale=1., c_scale=1., test_flag=False): # SOA out
        img = self.img_t if test_flag else self.img_s
        if prev_flag == -1:
            mode1_res = np.load(self.root_dir + "mode1_res.npy")
            mode1_res = np.transpose(mode1_res, [0, 2, 1, 3])
        elif prev_flag == 0:
            mode1_res = self._expmode(2, img, True, savedir=f'mode1_sim0.npy', test_flag=test_flag) * s_scale
        else:
            mode1_res = self._expmode(2, img, True, savedir=f'mode1_sim1.npy', test_flag=test_flag) * s_scale
        self._change_datafile(data_dict)
        mode2_sim = self._expmode(3, mode1_res, True, savedir=f'mode2_sim{int(test_flag)}_{int(sys)}.npy', test_flag=test_flag)
        mode2_res = np.transpose(mode2_res, [0, 2, 1, 3, 4]) * self.scale * c_scale
        self._plot_re(mode2_res, mode2_sim, self.root_dir + f'mode2_re{int(test_flag)}_{int(sys)}.png')
        self._expmode(4, mode2_sim, True, True, 'mode2 sim', test_flag=test_flag)
        self._expmode(4, mode2_res, True, True, 'mode2 res', test_flag=test_flag)
        return
    
    
    def mode4_eval(self, mode4_res, data_dict, prev_flag=-1, sys=False, test_flag=False): # LOOP out
        if data_dict is not None:
            self._change_datafile(data_dict)
            self.scale = scio.loadmat(data_dict['CoeffFile'])['coeff'][2, 0]
        if prev_flag == -1:
            mode2_res = np.load(self.root_dir + f"mode2_res{int(test_flag)}.npy")
            mode2_res = np.transpose(mode2_res, [0, 2, 1, 3, 4]) * self.scale
        elif prev_flag == 0:
            mode2_res = np.load(self.root_dir + f"mode2_sim{int(test_flag)}_0.npy")
        else:
            mode2_res = np.load(self.root_dir + f"mode2_sim1{int(test_flag)}_1.npy")
        mode4_sim = self._expmode(4, mode2_res, True, savedir=f'mode4_sim{int(test_flag)}_{int(sys)}.npy', test_flag=test_flag)
        mode4_sim = mode4_sim[..., -5:]
        mode4_res = mode4_res[..., -1, :]
        mode4_res = np.transpose(mode4_res, [0, 2, 1, 3])
        self._plot_re(mode4_res, mode4_sim, self.root_dir + f'mode4_re{int(test_flag)}_{int(sys)}.png')
        self._expmode(5, mode4_sim, True, True, 'mode4 sim', test_flag=test_flag)
        self._expmode(5, mode4_res, True, True, 'mode4 res', test_flag=test_flag)
        return
    
    
    def mode5_eval(self, mode5_res, data_dict, sys, test_flag): # LOOP out
        img = self.img_t if test_flag else self.img_s
        self._change_datafile(data_dict)
        mode5_sim = self._expmode(-1, img, True, test_flag=test_flag,
                                  savedir=f'mode5_sim{int(sys)}_{int(test_flag)}.npy')
        self._plot_re(mode5_res, mode5_sim, self.root_dir + 'mode5_re.png')
        self._expmode5_testall(mode5_res, test_flag)
        return
    
    
    def _change_datafile(self, data_dict_):
        if data_dict_ is None:
            return
        data_dict = deepcopy(data_dict_)
        for k in ['PropagationFile', 'SOAFile', 'CoeffFile']:
            if k not in data_dict or data_dict[k] is None:
                data_dict[k] = self.params.core['data'][k]
        if data_dict['load_checkpoint_dir'] is not None:
            self.load_checkpoint(data_dict['load_checkpoint_dir'])
        if data_dict['PostWeightFile'] is not None:
            i2o_state = torch.load(data_dict['PostWeightFile'])
            self.network.i2o.load_state_dict(i2o_state)
        data_dict.pop('load_checkpoint_dir')
        data_dict.pop('PostWeightFile')
        self.network.Change(**data_dict)
        
        
    def _plot_re(self, res, sim, figdir):
        re = np.sqrt(np.mean(np.square(sim-res)))/np.mean(res)
        plt.scatter(sim.flatten(), res.flatten(), s=1)
        _ = plt.title(f'{re}')
        plt.plot([np.min(sim), np.max(sim)], [np.min(sim), np.max(sim)], color="red")
        plt.savefig(figdir)
        print(f're: {re}')
        plt.close()
        return 
    
        
    def _expmode1_DMDin(self, i_max, dataloader, prefix, suffix, refresh=False):
        self.network.Change(ExpMode=1, NoiseStd=[0, 0, 0])
        if refresh:
            labels, img_all = [], []
            log = AverageMeterList(2)
            for index_dataset, samples in enumerate(tqdm(dataloader, leave=False, desc='Val')):
                torch.cuda.empty_cache()
                sample, label = samples
                sample = sample.to(self.device)
                label = label.to(self.device)
                output, rm_x = self.network(sample)
                loss = self.criterian(output, label)
                accuracy = self.evaluate(output, label)
                log.update([loss.cpu().detach().numpy(),  
                            accuracy.cpu().detach().numpy()])
                
                if index_dataset < i_max:
                    img_all.append(rm_x) # b, 10, 130, 130
                    labels.append(label)
                    
            tqdm.write(f'{prefix}: {log}')
            ls, bs = labels[-1].shape[0], self.params.batch_size
            if ls < bs:
                labels[-1] = torch.concat([labels[-1], torch.zeros(bs - ls, device=self.device) - 1], dim=0)
                img_all[-1] = torch.concat([img_all[-1], 
                                            torch.zeros(bs - ls, *img_all[-1].shape[1:], device=self.device)], dim=0)
            labels = torch.stack(labels, dim=0).type(torch.long)
            img_all = torch.stack(img_all, dim = 0) # (7, b, od, fm, 26, 26)
            
            torch.save(labels, self.root_dir + f"0_label{suffix}.pt")
            torch.save(img_all, self.root_dir + f"0_img_all{suffix}.pt")
        else:
            labels = torch.load(self.root_dir + f"0_label{suffix}.pt")
            img_all = torch.load(self.root_dir + f"0_img_all{suffix}.pt")
        return labels, img_all, img_all.cpu().detach().numpy()
    

    def _expmode(self, expmode, prev_res, refresh=False, print_flag=True,
                 eval_flag=False, prefix='Val', savedir=None, test_flag=False):
        if test_flag:
            img, label = self.img_t, self.label_t
        else:
            img, label = self.img_s, self.label_s
        if eval_flag or refresh or os.path.exists(self.root_dir + savedir):
            self.network.Change(expmode)
            log = AverageMeterList(2)
            xws = []
            if isinstance(prev_res, np.ndarray):
                prev_res = torch.from_numpy(prev_res).to(self.device)
            for idx in range(img.shape[0]):
                output, xw = self.network(prev_res[idx])
                loss = self.criterian(output, label[idx])
                accuracy = self.evaluate(output, label[idx])
                log.update([loss.cpu().detach().numpy(), accuracy.cpu().detach().numpy()])
                xws.append(xw)  
            if print_flag:
                print(f'{prefix}: {log}')
            if eval_flag:
                return
            xws = torch.stack(xws, dim=0)
            xws = xws.cpu().detach().numpy()
            np.save(self.root_dir + savedir, xws)
        else:
            xws = np.load(self.root_dir + savedir)
        return xws
    
    
    def _expmode5_testall(self, mode5_res, test_flag=False):
        if test_flag:
            img, label = self.img_t, self.label_t
        else:
            img, label = self.img_s, self.label_s
        self.network.Change(5)
        log = AverageMeterList(2)
        mode5_res = torch.from_numpy(mode5_res).to(self.device)
        for idx in range(img.shape[0]):
            output, _ = self.network(mode5_res[idx])
            loss = self.criterian(output, label[idx])
            accuracy = self.evaluate(output, label[idx])
            log.update([loss.cpu().detach().numpy(), accuracy.cpu().detach().numpy()])  
        print(f'mdoe5 res: {log}')
        
        
class Myloss:
    def __init__(self, regu=1, regu2=1, regu3=1, thers=1, sp = 0) -> None:
        self.CE = nn.CrossEntropyLoss()
        self.regu = regu
        self.regu2 = regu2
        self.regu3 = regu3
        self.thers = thers
        self.relu = nn.ReLU()
        self.low = 0.05
        self.high = 0.25
        self.sp = sp
        self.sparse = lambda x : x * (x < 1) + (5 - x) / 4 * (x >= 1)
    
    
    def __call__(self, output, label, weight=None):
        output, x, y_log = output
        loss = self.CE(output, label)
        if self.regu != 0:
            # regu_loss = torch.sqrt(torch.mean(torch.square(self.relu(y_log - self.thers))))
            regu_loss = torch.mean(self.relu(y_log - self.thers))
            loss = loss + self.regu * regu_loss
        if self.regu2 != 0: # y_log b, od, 50
            y_log = torch.mean(torch.mean(y_log, dim=2), dim=0)
            regu_loss2 = torch.sum(torch.relu(self.thers * 0.5 - y_log))
            
            loss = loss + self.regu2 * regu_loss2
        if self.regu3 != 0:
            x = x.flatten(1, 2)
            regu_loss3 = torch.sqrt(torch.mean(torch.square(x - torch.mean(x, dim=1, keepdim=True)), dim=0))
            regu_loss3 = torch.sum(torch.relu(regu_loss3-0.03), dim=0)
            loss = loss + self.regu3 * regu_loss3
        if self.sp != 0:
            sparse_loss = torch.mean(self.sparse(torch.clamp(weight.abs().flatten(), 0, 5)))
            loss = loss + self.sp * sparse_loss
        return loss

        

def Myeval(output, label):
    if isinstance(output, tuple):
        output, _, _ = output
    top_n, top_i = output.topk(1)
    flag = label != -1
    return (top_i[flag, 0] == label[flag]).float().mean()

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    params = SimParams()
    agent = SimAgent(params)
    agent.train()
    pass