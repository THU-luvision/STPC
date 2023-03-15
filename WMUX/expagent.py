from agent import EXPagent
from exputils import *
from parameter import *

def smooth(x):
    x = np.insert(x, [0, -1], [x[0], x[-1]])
    x1 = np.convolve(x, np.ones(3)/3, 'valid')
    x1[0], x1[-1] = x[0], x[-1]
    return x1

class FullSystem: 
    dmd_pos = [(1200, 600), (800, 400)]
    dmd_rmax = 250
    sample_rate = 16 / 3.567e-9 # 17 / 3.567e-9
    awg_max = [1.00, 0.090, 0.140, 0.070]
    bs = 10
    gw2 = generate_weight(delay=156, sep=437, peak_shape=np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    gw3 = generate_weight(delay=10, sep=437, peak_shape=np.array([ [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
    gw4 = generate_weight(delay=1097, sep=437, peak_shape=np.array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    gw5 = generate_weight(delay=197, sep=437, peak_shape=np.array([ [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    
    def __init__(self, savedir):
        self.rm = pyvisa.ResourceManager()
        self.awg4 = AWG7(self.rm, 'USB::0x1AB1::0x0645::DG7A24230xxxx::INSTR', ch=[1, 2, 3, 4], awg_max=self.awg_max) # four digits of the serial are masked for safty
        self.dmd = DMD(1)
        self.oscp = OSCP(self.rm, 'USB::0x0699::0x0530::C03xxxx::INSTR') # four digits of the serial are masked for safty
        self.laser = Laser()
        os.makedirs(f'./data/{savedir}', exist_ok=True)
        os.makedirs(f'./data/{savedir}/test', exist_ok=True)
        os.makedirs(f'./data/{savedir}/val', exist_ok=True)
        self.savedir = savedir
    
        self.laser.reset()
        self._awg_clk()
        
    def test(self):
        """ Test Device
        """
        self.dmd.put_circle(self.dmd_pos, [self.dmd_rmax, self.dmd_rmax])
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.92)
        self.laser.set_state(state=1, port=2, power=17.8, wav=1550.12)
        self._awg_tw(1)
        sleep(5)
        self._awg_sw1(1)
        sleep(10)
        self._awg_sw2(1)
        sleep(10)
        self._awg4_on()
        pass
    
    def calvi(self, port=1):
        """ Calibration of the Voltage to light Intensity conversion

        Args:
            port (int, optional): calibration ports(1 for laser-1, 2 for laser-2, 3 for VOA). Defaults to 1.
        """
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.92)
        self.laser.set_state(state=1, port=2, power=17.8, wav=1550.12)
        spec = 3.567e-9 / self.oscp.query_xspec() * np.arange(0, 3)[..., None]
        
        if port == 1 or port == 2:
            res, o, v2i = np.zeros((20, 20)), np.zeros((20, 20, 2)), np.zeros((20, 2))
            self._awg_tw(1)
            sleep(5)
            if port == 1:
                self.dmd.put_circle(self.dmd_pos, [self.dmd_rmax, 0])
                ss=[659631, 669919]
                sub_pos = (np.arange(957, 1030)[None, ...] + spec).astype(np.int_)
                data_src = 2
                self._awg_sw1(self.awg_max[1])
            else:
                self.dmd.put_circle(self.dmd_pos, [0, self.dmd_rmax])
                ss=[661477, 672119]
                sub_pos = (np.arange(883, 952)[None, ...] + spec).astype(np.int_)
                data_src = 3
                self._awg_sw2(self.awg_max[2])
            sleep(10)
            self._awg4_on()
            self.awg4.set_amp_off(ch=data_src, amp=1., offset=1./2)
            sleep(0.05)
            d = self.oscp.read_waveform(data_src, start_stop=ss)
            temp = np.mean(d[sub_pos], axis=1)
            alpha = np.array([1, temp[1] / temp[0] , temp[2] / temp[0]])
            
            for i, v in enumerate(tqdm(np.linspace(1, 0.001, 20))):
                self.awg4.set_amp_off(ch=data_src, amp=v, offset=v/2)
                sleep(0.05)
                d = self.oscp.read_waveform(data_src, start_stop=ss)
                v2i[i, :] = [v, np.mean(d[sub_pos[0, :]])]
            i_max = v2i[0, 1]
            v2i[:, 1] = smooth(v2i[:, 1] / i_max)
            
            for m, i in enumerate(tqdm(np.linspace(1, 0.001, 20))):
                v = np.interp(i, v2i[:, 1][::-1], v2i[:, 0][::-1])
                self.awg4.set_amp_off(ch=data_src, amp=v, offset=v/2)
                for n, r in enumerate(tqdm(np.linspace(1, 0, 20), leave=False)):
                    if port == 1:
                        self.dmd.put_circle(self.dmd_pos, [r*self.dmd_rmax, 0])
                    else:
                        self.dmd.put_circle(self.dmd_pos, [0, r*self.dmd_rmax])
                    sleep(0.05)
                    d = self.oscp.read_waveform(data_src, start_stop=ss)
                    res[m, n] = np.mean(d[sub_pos[0, :]])
                    o[m, n, :] = [i, r]
            scio.savemat(f'./data/{self.savedir}/{port}_res.mat', { 'res': res, 'o': o, 'alpha': alpha, 
                                                                    'v2i': v2i, 'i_max': i_max})
        else:
            ss=[661477, 672119]
            res = np.zeros((20, 2))
            self._awg_tw(self.awg_max[3])
            sleep(5)
            self._awg_sw2(1)
            self.dmd.put_circle(self.dmd_pos, [0, self.dmd_rmax])
            sleep(10)
            self._awg4_on()
            for i, v in enumerate(tqdm(np.linspace(1, 0.001, 20))):
                self.awg4.set_amp_off(ch=4, amp=v, offset=v/2)
                sleep(0.05)
                d = self.oscp.read_waveform(ch=3, start_stop=ss)
                res[i, :] = [v, np.mean(d[np.arange(883, 952)])]
            res[:, 1] = smooth(res[:, 1])
            scio.savemat(f'./data/{self.savedir}/{port}_res.mat', {'res': res})
        pass
    
    def calSOA(self):
        """SOA Calibration
        """
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.92)
        self.laser.set_state(state=1, port=2, power=17.8, wav=1550.12)
        d1 = scio.loadmat(f'./data/{self.savedir}/1_res_m.mat')
        res1 = np.stack([d1['o'][0, :, 1], d1['res'][0, :]], axis=1)
        d2 = scio.loadmat(f'./data/{self.savedir}/2_res_m.mat')
        res2 = np.stack([d2['o'][0, :, 1], d2['res'][0, :]], axis=1)
        res = np.zeros((20, 20, 2))
        o = np.zeros((20, 20, 2))
        ss1, ss2 = [699123, 709211], [700760, 711527]
        sub_pos1, sub_pos2 = np.arange(733, 855), np.arange(994, 1115)
        
        
        self._awg_tw(1)
        sleep(5)
        self._awg_sw1(1)
        sleep(10)
        self._awg_sw2(1)
        sleep(10)
        self._awg4_on()
        
        for i, o1 in enumerate(tqdm(np.linspace(res1[0, 1], 0, 20))):
            v1 = np.interp(o1, res1[:, 1][::-1], res1[:, 0][::-1])
            v1= np.clip(v1, 0, 1)
            for j, o2 in enumerate(tqdm(np.linspace(res2[0, 1], 0, 20), leave=False)):
                v2 = np.interp(o2, res2[:, 1][::-1], res2[:, 0][::-1])
                v2 = np.clip(v2, 0, 1)
                self.dmd.put_circle(self.dmd_pos, [v1*self.dmd_rmax, v2*self.dmd_rmax])
                sleep(0.05)
                d = self.oscp.read_waveform(2, start_stop=ss1)
                res[i, j, 0] = np.mean(d[sub_pos1])
                d = self.oscp.read_waveform(3, start_stop=ss2)
                res[i, j, 1] = np.mean(d[sub_pos2])
                o[i, j, :] = np.array([o1, o2])
        scio.savemat(f'./data/{self.savedir}/1_soares.mat', {'res': res[..., 0], 'o': o})
        scio.savemat(f'./data/{self.savedir}/2_soares.mat', {'res': res[..., 1], 'o': o})
        pass
    
    def validate_1(self, exp_num=1, split='val'): 
        """Experimental verification of the first layer

        Args:
            exp_num (int, optional): Flag indicating SOA modules(1 for SOA input, 2 for SOA output). Defaults to 1.
            split (str, optional): dataset split('val' or 'test'). Defaults to 'val'.
        """
        params = ExpParams()
        self.agent = EXPagent(params)
        y1, y2, tw, gt = self.agent._generate_exp_data_1(self.bs, sys=False, exp_num=exp_num, split=split)        
        y1_e, y1_o = y1
        y2_e, y2_o = y2
        
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.92)
        self.laser.set_state(state=1, port=2, power=17.8, wav=1550.12)
        self._awg_tw(tw, cal=False)
        sleep(5)
        self.dmd.put_circle(self.dmd_pos, [y1_o * self.dmd_rmax, y2_o * self.dmd_rmax])
        res = np.zeros((2, y1_e.shape[0], self.bs, 2))
        
        if exp_num == 1:
            sub_pos1, sub_pos2 = np.arange(667722, 667795), np.arange(669494, 669563)
        else:
            sub_pos1, sub_pos2 = np.arange(706834, 707078), np.arange(708937, 709127)
            
        spec = self.oscp.query_xspec() * self.sample_rate
        spec = (485 / spec * np.arange(0, 10))[:, None] + (7 / spec * np.arange(0, 2))[None, :]
        sub_pos1 = (sub_pos1[..., None, None] + spec[None, ...]).astype(np.int_) # N, 10, 2
        sub_pos2 = (sub_pos2[..., None, None] + spec[None, ...]).astype(np.int_)
        for i in range(y1_e.shape[0]):
            self._awg_sw1(y1_e[i], cal=False)
            sleep(10)
            self._awg_sw2(y2_e[i], cal=False)
            sleep(10)
            self._awg4_on()
            sleep(0.05)
            if i == 2 or i == 5 or split=='test':
                scio.savemat(f'./data/{self.savedir}/{split}/1_{exp_num}_full_{i}iter.mat', 
                             {'wfm2': self.oscp.read_waveform(2), 'wfm3': self.oscp.read_waveform(3)})
            d = self.oscp.read_waveform(2)
            res[0, i, :] = np.mean(d[sub_pos1], axis=0) # 10, 2
            d = self.oscp.read_waveform(3)
            res[1, i, :] = np.mean(d[sub_pos2], axis=0) # 10, 2
            print(f'{i} re: {np.sqrt(np.mean(np.square(gt[:, i]-res[:, i])))/np.mean(res[:, i])}')
        scio.savemat(f'./data/{self.savedir}/{split}/1_{exp_num}_net_res.mat', 
                     {'res': res, 'gt': gt, 'label': self.agent.label.cpu().detach().numpy()})
        self.agent._calculate_exp_res_1(res, exp_num=exp_num)
        print(f're: {np.sqrt(np.mean(np.square(gt-res)))/np.mean(res)}')
        pass
    
    def validate_2P(self, split='val'):
        """Experimental verification of the first layer

        Args:
            split (str, optional): dataset split('val' or 'test'). Defaults to 'val'.
        """
        params = ExpParams()
        self.agent = EXPagent(params)
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.92)
        self.laser.set_state(state=1, port=2, power=17.8, wav=1550.12)
        self.dmd.put_circle(self.dmd_pos, [self.dmd_rmax, self.dmd_rmax])
        y0_v, gt, tw = self.agent._generate_exp_data_2(self.savedir, split=split)
              
        res = np.zeros_like(gt) # 2, 12, 5, 2
        spec = 485 / self.sample_rate / self.oscp.query_xspec() * np.arange(0, 10)
        sub_pos = np.arange(655235, 655293)[None, ...] + \
                  np.array([[0], [7 / self.sample_rate / self.oscp.query_xspec()]])
        sub_pos = (spec[..., None, None] + sub_pos).astype(np.int_)
        self._awg_tw(tw, cal=False)
        sleep(5)
        for i in tqdm(range(y0_v.shape[1]), leave=False):
            self._awg_sw1(y0_v[0, i], cal=False) # 2, 12, 5, 3, 1
            sleep(10)
            self._awg_sw2(y0_v[1, i], cal=False)
            sleep(10)
            self._awg4_on()
            sleep(1)
            if i == 2 or i == 5 or split == 'test':
                scio.savemat(f'./data/{self.savedir}/{split}/2_full_{i}iter.mat', {'wfm': self.oscp.read_waveform(2)})
            d = self.oscp.read_waveform(2)
            res[i] = np.mean(d[sub_pos], axis=2)
            tqdm.write(f'{i} re: {np.sqrt(np.mean(np.square(gt[i]-res[i])))/np.mean(res[i])}')
        scio.savemat(f'./data/{self.savedir}/{split}/2_net_res.mat', 
                     {'res': res, 'gt': gt, 'label': self.agent.label})
        print(f're: {np.sqrt(np.mean(np.square(gt-res)))/np.mean(res)}')
        self.agent._calculate_exp_res_2(gt)
        self.agent._calculate_exp_res_2(res)
        return
    
    def _transform_sw(self, v):
        tmp0 = np.zeros((self.bs, 3, 1))
        tmp0[:, 0, :] = v
        return tmp0
    
    def _transform_tw(self, v):
        tmp0 = np.zeros((self.bs, 3, 2))
        tmp0[:, 0, :] = v
        return tmp0
    
    def _awg_clk(self):
        tmp1 = np.concatenate([np.zeros(5), np.ones(1000), np.zeros(1000)])
        self.awg4.arbitrary(tmp1, ch=1, sample_rate=self.sample_rate, wfm_name='mod_1')
    
    def _awg_sw1(self, tmp, amp=1, cal=True):
        tmp1 = self._transform_sw(tmp) if cal else tmp
        tmp1 = self.gw2(tmp1, amp=amp)
        self.awg4.arbitrary(tmp1, ch=2, sample_rate=self.sample_rate, wfm_name='mod_2')
    
    def _awg_tw(self, tmp, amp=1, cal=True):
        tmp0 = self._transform_tw(tmp)  if cal else tmp
        tmp0 = self.gw4(tmp0, amp=amp)
        self.awg4.arbitrary(tmp0, ch=4, sample_rate=self.sample_rate, wfm_name='mod_4')
        
    def _awg_sw2(self, tmp, amp=1, cal=True):
        tmp1 = self._transform_sw(tmp) if cal else tmp
        tmp1 = self.gw3(tmp1, amp=amp)
        self.awg4.arbitrary(tmp1, ch=3, sample_rate=self.sample_rate, wfm_name='mod_3')

    def _awg4_on(self, ch=[1, 2, 3, 4]):
        for c in ch:
            self.awg4.write_wait([f':SOURce{c}:CASSet:WAVeform mod_{c}'])
        self.awg4.set_state(state=[1]*len(ch), ch=ch, sync=True)
        
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # ma = FullSystem('1021_1')
    # ma.test()
    # ma.calvi(port=1)
    # ma.calvi(port=2)
    # ma.calvi(port=3)
    # ma.calSOA()
    
    # ma.validate_1(exp_num=1, split='val')
    # ma.validate_1(exp_num=2, split='val')
    # ma.validate_2P(split='val')
    
    # ma.validate_1(exp_num=1, split='test')
    # ma.validate_1(exp_num=2, split='test')
    # ma.validate_2P(split='test')
