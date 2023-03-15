from agent import EXPagent
from exputils import *
from parameter import *
from utils import generate_weight

def smooth(x):
    x = np.insert(x, [0, -1], [x[0], x[-1]])
    x1 = np.convolve(x, np.ones(3)/3, 'valid')
    x1[0], x1[-1] = x[0], x[-1]
    return x1

def get_local_mean(rawdata, sample_point, region=3):
    tmp = 0
    for i in range(-region//2, -region//2 + region):
        tmp += rawdata[sample_point + i]
    return tmp / float(region)

def dmd_point(ref_point, loop_spec):
    sample_point = np.round(loop_spec() * np.arange(990))
    sample_point = sample_point.reshape(33, 30)[:, :10] + ref_point # (33, 10)
    return sample_point.astype(int)

def soa_point(ref_point, loop_spec):
    sample_point = dmd_point(ref_point, loop_spec) # 33, 10
    sample_point = np.repeat(sample_point[..., None], 5, axis=2) # 33, 10, 5
    sample_point += (np.round(loop_spec() / 200. * 16) * np.arange(5)[None, None, :]).astype(int)
    # 200 loop sample point; 16 weight sample point
    return sample_point

def cloop_point(ref_point, loop_spec):
    sample_point1 =  np.round(loop_spec() / 200. * 16) * np.arange(5) # 5
    sample_point2 = ref_point[1] + np.round(loop_spec()) * np.arange(6) # 6 
    sample_point2 = sample_point2[:, None] + sample_point1[None, :] # 6, 5
    sample_point1 += ref_point[0]
    return sample_point1.astype(int), sample_point2.astype(int)

def loop_point(ref_point, loop_spec):
    sample_point = np.round(loop_spec() * np.arange(990))
    sample_point -= sample_point[9]
    sample_point = sample_point.reshape(33, 30)[:, :10, None] + np.array(ref_point)[None, None, :] # (33, 10, 5)
    return sample_point.astype(int) # 33, 5

class ExpBase:
    rm = pyvisa.ResourceManager()
    gw = generate_weight()
    loop_freq = 8038.
    delay = [0.001, 2.424e-3]
    laser_par = [[1550.12, 17.8], [1557.12, 15.8]]
    
    def __init__(self, exp_id):
        self.dmd = DMD(1000)
        self.awg1 = AWG4(self.rm, 'USB::0x1AB1::0x0641::DG4E24260xxxx::INSTR', ch=[1, 2]) # four digits of the serial are masked for safty
        self.awg2 = AWG4(self.rm, 'USB::0x1AB1::0x0641::DG4E24250xxxx::INSTR', ch=[1, 2]) # four digits of the serial are masked for safty
        self.awg3 = AWG(self.rm, 'USB::0x1AB1::0x0642::DG1ZA23040xxxx::INSTR', ch=[1, 2]) # four digits of the serial are masked for safty
        self.oscp = OSCP(self.rm, 'USB::0x0699::0x052C::C03xxxx::INSTR') # four digits of the serial are masked for safty
        self.laser = Laser()
        self.reset()
        self.root_dir = f"data\\{exp_id}\\"
        os.makedirs(self.root_dir, exist_ok=True)
        
    def reset(self):
        self.dmd.reset()
        self.laser.reset()
        self.awg1.reset()
        self.awg2.reset()
    
    def test(self):
        self._laser_state(True, 1)
        self.dmd.put_white()
        self._trig('set')
        self._dmdsignal()
        self._trig('on')
        return

    def loop_spec(self):
        return 1 / self.loop_freq / self.oscp.query_xspec()

    def _trig(self, mode):
        if mode == 'set':
            self.awg3.set_state(ch=1, state=0)
            self.awg3.square(ch=1, freq=803.8, amp=5., offset=2.5)
            self.awg3.burst(ch=1, nc=1, period=0.15, idle='BOTTOM')
            self.awg2.set_state(ch=1, state=0)
            self.awg2.square(ch=1, freq=803.8, amp=5., offset=2.5)
            self.awg2.burst(ch=1, nc=1, source='EXT')
            self.awg3.set_state(ch=1, state=1)
        elif mode == 'on':
            self.awg2.set_state(ch=1, state=1)
        elif mode == 'off':
            self.awg2.set_state(ch=1, state=0)
    
    def _dmdsignal(self):
        self.awg1.reset(ch=1)
        self.awg1.square(ch=1, freq=8038, amp=5., offset=2.5)
        self.awg1.burst(ch=1, nc=1000, source='EXT', delay=self.delay[0])
        self.awg1.set_state(1, ch=1, burst=True)
        
    def _soasignal(self, weight, amp=5):
        self.awg2.reset(ch=2)
        self.awg2.arbitrary(weight, ch=2, sample_rate=1607600, amp=amp, offset=amp/2.)
        self.awg2.burst(ch=2, source='EXT', nc=33, idle='BOTTOM', delay=self.delay[1])
        self.awg2.set_state(state=1, ch=2)
    
    def _voasignal(self, v, reset=False):
        if reset:
            self.awg1.reset(ch=2)
        self.awg1.dc(ch=2, offset=v)
        if reset:
            self.awg1.set_state(state=1, ch=2)
            
    def _putwhites(self):
        tmp = np.zeros((33, 30, 26, 26), dtype=int)
        tmp[:, :10] = 255
        tmp = np.concatenate([np.zeros((10, 26, 26), dtype=int), tmp.reshape(-1, 26, 26)], axis=0) # 1000, 26, 26
        self.dmd.put_imgs(tmp)
        
    def _laser_state(self, state, port): # state T/F, port 1/2
        if state:
            self.laser.set_state(state=1, port=port, 
                                 power=self.laser_par[port-1][1], wav=self.laser_par[port-1][0])
        else:
            self.laser.set_state(state=0, port=port)

class FullSystem(ExpBase):
    sys_i, test_i = 2, 1
    laser_par = [[1550.12, 17.8], [1557.12, 15.8]]
    def __init__(self, force_i=False, force_w=False):
        np.random.seed(25)
        torch.cuda.manual_seed_all(25)
        torch.manual_seed(25)
        params = ExpParams()
        
        super().__init__(params.exp_id)
        self.root_dir = f"data\\{params.exp_id}\\"
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.root_dir + 'model', exist_ok=True)
        self.agent = EXPagent(params)
        self._load_params(params)
        self._load_imgw(force_i, force_w)
        

    def test(self):
        """Test Device
        """
        self.dmd.put_white()
        self.laser.set_state(state=1, port=1, power=17.8, wav=1550.12)
        return
    
    def mode1_dmd(self, ref=False, exp_flag=True, ref_point=0, data_dict=None, sys=False, c_scale=1.):
        """validate DMD output

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            exp_flag (bool, optional): flag of conducting experiments or loading historical data. Defaults to True.
            ref_point (int, optional): reference point got in MATLAB. Defaults to 0.
            data_dict (dict or None, optional): system learning data. Defaults to None.
            sys (bool, optional): flag of system learning. Defaults to False.
            c_scale (float, optional): calibration scale. Defaults to 1..
        """
        self._laser_state(True, 1)
        data_src = 2
        if exp_flag:
            i_max = self.sys_i
            outd, batch_size, fm = self.outd, self.bs, self.fm
            sample_point = dmd_point(ref_point, self.loop_spec)
            mode1_res = np.zeros((i_max, outd, batch_size, fm))
            self._trig('set')
            self._dmdsignal()
            for i in range(i_max):
                for d in range(outd):
                    self._trig('off')
                    self._dmd_put(i, d, batch_size)
                    self._trig('on')
                    sleep(1)
                    if ref:
                        scio.savemat(self.root_dir + 'mode1_full.mat', 
                                     {'wfm': self.oscp.read_waveform(data_src)}, do_compression=True)
                        return
                    rawdata = self.oscp.read_waveform(ch=data_src)
                    mode1_res[i, d] = rawdata[sample_point]
            np.save(self.root_dir + "mode1_res.npy", mode1_res)
        else:
            mode1_res = np.load(self.root_dir + "mode1_res.npy")
        self.agent.mode1_eval(mode1_res, data_dict, sys, c_scale) # print re(sim&res, res1+sim2&res) accuracy loss
        return
         
    def mode2_soa(self, ref=False, exp_flag=True, ref_point=0, prev_flag=-1, data_dict=None, sys=False, c_scale=1., test_flag=False, s_scale=1.):
        """validate SOA output

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            exp_flag (bool, optional): flag of conducting experiments or loading historical data. Defaults to True.
            ref_point (int, optional): reference point got in MATLAB. Defaults to 0.
            prev_flag (int, optional): sources of simulation result generation(-1 for experimental data, 0 for simulated data). Defaults to -1.
            data_dict (dict or None, optional): system learning data. Defaults to None.
            sys (bool, optional): flag of system learning. Defaults to False.
            c_scale (float, optional): calibration scale. Defaults to 1..
            test_flag (bool, optional): flag of dataset type(True for test dataset, False for train dataset). Defaults to False.
            s_scale (float, optional): calibration scale2. Defaults to 1..
        """
        self._laser_state(True, 1)
        data_src = 2
        if ref:
            self._dmdsignal()
            self._trig('set')
            self._putwhites()
            self._soasignal(self.weight_ref[0])
            self._trig('on')
            sleep(1.)
            scio.savemat(self.root_dir + 'mode2_full.mat',
                         {'wfm': self.oscp.read_waveform(data_src)}, do_compression=True)
            return
        
        if exp_flag:
            i_max = self.test_i if test_flag else self.sys_i
            outd, batch_size, fm, hs = self.outd, self.bs, self.fm, self.hs
            sample_point = soa_point(ref_point, self.loop_spec)
            mode2_res = np.zeros((i_max, outd, batch_size, fm, hs))
            self._trig('set')
            self._dmdsignal()
            for i in range(i_max):
                for d in range(outd):
                    self._trig('off')
                    self._dmd_put(i, d, batch_size, test=test_flag)
                    self._soasignal(self.weight[d])
                    self._trig('on')
                    
                    sleep(1)
                    rawdata = self.oscp.read_waveform(ch=data_src)
                    mode2_res[i, d] = rawdata[sample_point]
            np.save(self.root_dir + f"mode2_res{int(test_flag)}.npy", mode2_res)
        else:
            mode2_res = np.load(self.root_dir + f"mode2_res{int(test_flag)}.npy")
        self.agent.mode2_eval(mode2_res, data_dict, prev_flag, sys, s_scale, c_scale, test_flag) # print re(res1+sim2&sim)
        return
    
    def mode3_cloop(self, ref=False, ref_point=[35335, 36576], w_max=5, pow2=None):
        """calibration loop

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            ref_point (list, optional): reference point got in MATLAB. Defaults to [35335, 36576].
            w_max (int, optional): upper limit of SOA weight. Defaults to 5.
            pow2 (float or None, optional): power of laser-2(None for no changes). Defaults to None.
        """
        if pow2 is not None:
            self.laser_par[1][1] = pow2
        self._laser_state(True, 1)
        self._laser_state(True, 2)
        
        data_src =  [2, 3]
        mode3_res1, mode3_res2 = np.zeros((10, 5)), np.zeros((10, 6, 5))
        weight_ref = self.weight_ref * (w_max / 5.)
        sp1, sp2 = cloop_point([ref_point[0], ref_point[1]], self.loop_spec)
        
        self._trig('set')
        self._dmdsignal()
        self._putwhites()
        for i in range(10):
            self._trig('off')
            self._soasignal(weight_ref[i])
            self._trig('on')
            sleep(0.5)
            if ref:
                scio.savemat(self.root_dir + 'mode3_full.mat', 
                            {'wfm1': self.oscp.read_waveform(ch=data_src[0]), 
                             'wfm2': self.oscp.read_waveform(ch=data_src[1]),
                             'scale': self.agent.scale}, do_compression=True)
                return 
            rawdata1 = self.oscp.read_waveform(ch=data_src[0])
            rawdata2 = self.oscp.read_waveform(ch=data_src[1])
            mode3_res1[i] = rawdata1[sp1]
            mode3_res2[i] = rawdata2[sp2]
        scio.savemat(self.root_dir + 'mode3_res.mat', 
                    {'rx2': mode3_res1, 'rx3': mode3_res2}, do_compression=True)
        
        self._trig('off')
        self._soasignal(weight_ref[0])
        self._trig('on')
        return
    
    def mode4_loop(self, ref=False, ref_point=[0, 0, 0, 0, 0], exp_flag=True, prev_flag=-1, data_dict=None, sys=False, test_flag=False):
        """validate loop

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            ref_point (list, optional): reference point got in MATLAB. Defaults to [0, 0, 0, 0, 0].
            exp_flag (bool, optional): flag of conducting experiments or loading historical data. Defaults to True.
            prev_flag (int, optional): sources of simulation result generation(-1 for experimental data, 0 for simulated data). Defaults to -1.
            data_dict (dict or None, optional): system learning data. Defaults to None.
            sys (bool, optional): flag of system learning. Defaults to False.
            test_flag (bool, optional): flag of dataset type(True for test dataset, False for train dataset). Defaults to False.
        """
        self._laser_state(True, 1)
        self._laser_state(True, 2)
        data_src = 3
        if exp_flag:
            i_max = self.test_i if test_flag else self.sys_i
            outd, batch_size, fm, hs = self.outd, self.bs, self.fm, self.hs
            
            mode4_res = np.zeros((i_max, outd, batch_size, fm, hs))
            sample_point = loop_point(ref_point, self.loop_spec) # bs, fm, hs
            self._trig('set')
            self._dmdsignal()
            for i in range(i_max):
                for d in range(self.outd):
                    self._trig('off')
                    self._dmd_put(i, d, batch_size, test=test_flag)
                    self._soasignal(self.weight[d], amp=5.)
                    self._trig('on')
                    
                    sleep(1)
                    if ref:
                        scio.savemat(self.root_dir + 'mode4_full.mat',
                                     {'wfm': self.oscp.read_waveform(data_src)}, do_compression=True)
                        return
                    rawdata = self.oscp.read_waveform(ch=data_src)
                    mode4_res[i, d] = rawdata[sample_point]
            np.save(self.root_dir + f"mode4_res{int(test_flag)}.npy", mode4_res)
        else:
            mode4_res = np.load(self.root_dir + f"mode4_res{int(test_flag)}.npy")
        self.agent.mode4_eval(mode4_res, data_dict, prev_flag, sys, test_flag)
    
    def mode5_test(self, ref=False, ref_point1=0, ref_point2=[0, 0, 0, 0, 0], exp_flag=True, prev_flag=-1, data_dict=None, sys=False, test_flag=False):
        """validate entire process

        Args:
            ref (bool, optional):  flag of reference model. Defaults to False.
            ref_point1 (int, optional): reference point got in MATLAB. Defaults to 0.
            ref_point2 (list, optional): reference point got in MATLAB. Defaults to [0, 0, 0, 0, 0].
            exp_flag (bool, optional): flag of conducting experiments or loading historical data. Defaults to True.
            prev_flag (int, optional): sources of simulation result generation(-1 for experimental data, 0 for simulated data). Defaults to -1.
            data_dict (dict or None, optional): system learning data. Defaults to None.
            sys (bool, optional): flag of system learning. Defaults to False.
            test_flag (bool, optional): flag of dataset type(True for test dataset, False for train dataset). Defaults to False.
        """
        self._laser_state(True, 1)
        self._laser_state(True, 2)
        data_src = [2, 3]
        if exp_flag:
            i_max = self.test_i if test_flag else self.sys_i
            outd, batch_size, fm, hs = self.outd, self.bs, self.fm, self.hs
            mode2_res = np.zeros((i_max, outd, batch_size, fm, hs))
            mode4_res = np.zeros((i_max, outd, batch_size, fm, hs))
            sp1 = soa_point(ref_point1, self.loop_spec)
            sp2 = loop_point(ref_point2, self.loop_spec) # bs, fm, hs
            self._trig('set')
            self._dmdsignal()
            for i in range(i_max):
                for d in range(self.outd):
                    self._trig('off')
                    self._dmd_put(i, d, batch_size, test=test_flag)
                    self._soasignal(self.weight[d], amp=5.)
                    self._trig('on')
                    
                    sleep(1)
                    if ref:
                        scio.savemat(self.root_dir + 'mode4_full.mat',
                                     {'wfm': self.oscp.read_waveform(data_src)}, do_compression=True)
                        return
                    rawdata1 = self.oscp.read_waveform(ch=data_src[0])
                    rawdata2 = self.oscp.read_waveform(ch=data_src[1])
                    mode2_res[i, d] = rawdata1[sp1]
                    mode4_res[i, d] = rawdata2[sp2]
            np.save(self.root_dir + f"mode2_res{int(test_flag)}.npy", mode2_res)
            np.save(self.root_dir + f"mode4_res{int(test_flag)}.npy", mode4_res)
        else:
            mode2_res = np.load(self.root_dir + f"mode2_res{int(test_flag)}.npy")
            mode4_res = np.load(self.root_dir + f"mode4_res{int(test_flag)}.npy")
        print('mode2_res:')
        self.agent.mode2_eval(mode2_res, data_dict, prev_flag, sys, 1., 1., test_flag)
        print('mode4 res:')
        self.agent.mode4_eval(mode4_res, data_dict, prev_flag, sys, test_flag)       
                
    
    def _load_params(self, params):
        self.bs = params.batch_size
        c = params.core['network']
        self.fm = c['input_size'] * c['frame_num']
        self.os, self.hs = c['output_size'], c['hidden_size']
        self.outd = c['output_dim']
        
    def _load_imgw(self, force_i, force_w):
        img_alls, img_allt = self.agent.generate_DMDimg(self.sys_i, self.test_i, force_i)
        self.img_alls = np.transpose(img_alls, [0, 2, 1, 3, 4, 5]) # (2, od, b, fm, 26, 26)
        self.img_allt = np.transpose(img_allt, [0, 2, 1, 3, 4, 5]) # (7, od, b, fm, 26, 26)
        self.weight, self.weight_ref = self.agent.generate_weight(force_w)# od, 4500; 10, 4500
    
    def _dmd_put(self, i, d, bs, test=False):
        img_all = self.img_allt if test else self.img_alls
        tmp = np.concatenate([img_all[i, d], np.zeros((bs, 20, 26, 26))], 
                            axis=1).reshape(-1, 26, 26) # b, 10+20, 26, 26 --> 990, 26, 26
        tmp = np.concatenate([np.zeros((10, 26, 26)), tmp], axis=0) # 1000, 26, 26
        tmp = tmp.astype(np.uint8) * 255
        self.dmd.put_imgs(tmp)
    
    
class Calbration(ExpBase):    
    def __init__(self, exp_id, refresh=False):
        super().__init__(exp_id + '_C')
        self._generate_DMDimg(refresh)

    def mode0_POScal(self, rx, ry):
        """calbration the position of square on the DMD

        Args:
            rx (float): x-coordinate of the square
            ry (float): y-coordinate of the square

        Returns:
            np.array: DMD image
        """
        self._laser_state(True, 1)
        tmp = self.dmd.set_position(rx=rx, ry=ry)
        self.dmd.put_img(tmp, transform=False)
        return tmp
    
    def mode1_DMDcal(self, ref=False, ref_point=0):
        self._laser_state(True, 1)
        data_src = 2
        mode1_res = np.zeros((10, 33, 10))
        sample_point = dmd_point(ref_point, self.loop_spec)
        self._trig('set')
        self._dmdsignal()
        
        for i in tqdm(range(10), leave=False):
            self._trig('off')
            tmp = np.concatenate([self.img_all[i], np.zeros((33, 20, 26, 26))], 
                            axis=1).reshape(-1, 26, 26) # b, 10+20, 26, 26 --> 990, 26, 26
            tmp = np.concatenate([np.zeros((10, 26, 26)), tmp], axis=0) # 1000, 26, 26
            self.dmd.put_imgs(tmp)
            self._trig('on')
            sleep(0.5)
            if ref:
                scio.savemat(self.root_dir + 'mode1_full.mat',
                             {'wfm': self.oscp.read_waveform(ch=data_src)}, do_compression=True)
                return

            rawdata = self.oscp.read_waveform(ch=data_src)
            mode1_res[i] = get_local_mean(rawdata, sample_point)
        np.save(self.root_dir + 'mode1_res.npy', mode1_res)
        return mode1_res
    
    def mode2_soa(self, ref=False, ref_point=0, out_flag=False, v_range=[1.28, 3.0], yscale=None):
        """SOA calibration

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            ref_point (int, optional): reference point got in MATLAB. Defaults to 0.
            out_flag (bool, optional): flag of calibrating SOA inputs or outputs. Defaults to False.
            v_range (list, optional): voltage range of VOA. Defaults to [1.28, 3.0].
            yscale (float or None, optional): y scale of OSCP. Defaults to None.
        """
        self._laser_state(True, 1)
        data_src = 2
        mode2_res = np.zeros((3, 33, 20)) if out_flag else np.zeros((34, 2))
        
        self._voasignal(v_range[0], True)
        self._trig('set')
        self._dmdsignal()
        self._putwhites()
        self._trig('off')
        
        if not out_flag:
            self._trig('on')
            sleep(1.)
            vs = np.linspace(v_range[0], v_range[1], 34)[:33]
            for i in tqdm(range(33), leave=False):
                self._voasignal(vs[i])
                if ref:
                    scio.savemat(self.root_dir + 'mode2_full0.mat',
                             {'wfm': self.oscp.read_waveform(ch=data_src)}, do_compression=True)
                    return
                sleep(0.5)
                rawdata = self.oscp.read_waveform(ch=data_src, start_stop=[ref_point-10, ref_point+10])
                mode2_res[i, 0] = np.mean(rawdata)
                mode2_res[i, 1] = vs[i]
            self.laser.reset()
            sleep(0.5)
            rawdata = self.oscp.read_waveform(ch=data_src, start_stop=[ref_point-10, ref_point+10])
            mode2_res[-1, 0] = np.mean(rawdata)
        else:
            sample_point = soa_point(ref_point, self.loop_spec)[0]# 33, 10, 5 --> 10, 5
            w_ = np.linspace(5, 0, 21)[:20].reshape(2, 10) # 10
            weight_ref = w_[..., None].repeat(repeats=5, axis=2) # 2, 10, 5
            weight_ref = self.gw(weight_ref) # 2, 5000
            
            i2v = np.load(self.root_dir + f'mode2_res0.npy') # 34, 2
            i2v = i2v[:33, :] - i2v[-1, :]
            is_ = np.linspace(i2v[0, 0], 0, 34)[:33]
            vs = np.interp(is_, i2v[:, 0][::-1], i2v[:, 1][::-1])
            
            for j in range(2):
                if yscale is not None:
                    self.oscp.set_yscale(ch=data_src, scale=yscale[j])
                
                self._trig('off')
                self._soasignal(weight_ref[j])
                self._trig('on')
                sleep(1.)
                for i in tqdm(range(33), leave=False):
                    self._voasignal(vs[i])
                    sleep(0.1)
                    if ref:
                        scio.savemat(self.root_dir + f'mode2_full1.mat', 
                                    {'wfm': self.oscp.read_waveform(ch=data_src)}, do_compression=True)
                        return
                    rawdata = self.oscp.read_waveform(ch=data_src)
                    mode2_res[2, i, (j * 10):(j * 10 + 10)] = np.mean(get_local_mean(rawdata, sample_point), axis=1)
            mode2_res[0, :, :] = is_[:, None] # 33, 1
            mode2_res[1, :, :] = w_.ravel()[None, :] # 1, 20
        scio.savemat(self.root_dir + f'mode2_res{int(out_flag)}.mat', {'res': mode2_res})
        np.save(self.root_dir + f'mode2_res{int(out_flag)}.npy', mode2_res)
        if yscale is not None:
            self.oscp.set_yscale(ch=data_src, scale=yscale[0])
        return mode2_res
    
    def mode3_cloop(self, ref=False, ref_point=[35335, 36576], w_max=5, v_range=[1.2, 3.]):
        """calibration loop

        Args:
            ref (bool, optional): flag of reference model. Defaults to False.
            ref_point (list, optional): reference point got in MATLAB. Defaults to [35335, 36576].
            w_max (int, optional): upper limit of SOA weight. Defaults to 5.
            v_range (list, optional): voltage range of VOA. Defaults to [1.2, 3.
            ].
        """
        self._laser_state(True, 1)
        self._laser_state(True, 2)
        
        data_src, ss =  [2, 3], [[34164, 34824], [35391, 43919]]
        mode3_res1, mode3_res2 = np.zeros((10, 5)), np.zeros((10, 6, 5))
        weight_ref = self._weight_ref() * (w_max / 5.)
        sp1, sp2 = cloop_point([ref_point[0] - ss[0][0], ref_point[1] - ss[1][0]], self.loop_spec)
        
        self._voasignal(v_range[0], True)
        self._trig('set')
        self._dmdsignal()
        self._putwhites()
        for i in range(10):
            self._trig('off')
            self._soasignal(weight_ref[i])
            self._trig('on')
            sleep(0.1)
            if ref:
                scio.savemat(self.root_dir + 'mode3_full.mat', 
                            {'wfm1': self.oscp.read_waveform(ch=data_src[0]), 
                             'wfm2': self.oscp.read_waveform(ch=data_src[1])}, do_compression=True)
                return 
            rawdata1 = self.oscp.read_waveform(ch=data_src[0], start_stop=ss[0])
            rawdata2 = self.oscp.read_waveform(ch=data_src[1], start_stop=ss[1])
            mode3_res1[i] = rawdata1[sp1]
            mode3_res2[i] = rawdata2[sp2]
        scio.savemat(self.root_dir + 'mode3_res.mat', 
                    {'rx2': mode3_res1, 'rx3': mode3_res2}, do_compression=True)
        
        self._trig('off')
        self._soasignal(weight_ref[0])
        self._trig('on')
        return
        
    def _generate_DMDimg(self, refresh):
        if refresh or not os.path.exists(self.root_dir + '0_img_all.npy'):
            self.img_all = []
            np.random.seed(20)
            rawimg = np.load('./data/subimg.npy') # 5000, 26, 26
            order = np.arange(5000)
            np.random.shuffle(order)
            for i in range(10):
                for j in range(10):
                    for k in range(33):
                        img = np.random.randint(0, 255, (26, 26))
                        thres = (0.43 + 0.03 * j) * 255
                        img[img > thres] = 255
                        img[img <= thres] = 0
                        if i < 2:
                            img = np.bitwise_and(img, 255 - rawimg[order[j * 33 + k]])
                        self.img_all.append(img)
            self.img_all = np.stack(self.img_all, axis=0).reshape(10, 33, 10, 26, 26)
            np.save(self.root_dir + '0_img_all.npy', self.img_all)
        else:
            self.img_all = np.load(self.root_dir + '0_img_all.npy')
        return 
    
    def _weight_ref(self):
        tmp = np.linspace(5, 0, 11)[:10] # 10
        weight_ref = np.zeros((10, 10, 5))
        weight_ref[:, 0, :] = tmp[:, None]
        weight_ref = self.gw(weight_ref) # 10, 5000
        return weight_ref

if __name__ == '__main__':
    # %%
    # calibration
    agent = Calbration('0302_1')
    
    tmp = agent.mode0_POScal(250, 150)
    
    agent.mode1_DMDcal(ref=True)
    agent.mode1_DMDcal(ref_point=551800)
    
    v_range = [2.63, 4.31]
    agent.mode2_soa(ref=True, v_range=v_range)
    agent.mode2_soa(ref=False, v_range=v_range, ref_point=34524)
    
    agent.mode2_soa(ref=True, v_range=v_range, out_flag=True)
    agent.mode2_soa(ref=False, v_range=v_range, out_flag=True, ref_point=34325, yscale=[70e-3, 35e-3])
    
    agent.mode3_cloop(ref=True, v_range=v_range, w_max=5.)
    agent.mode3_cloop(ref=False, ref_point=[34337, 35575],  w_max=5., v_range=v_range)
    
    # %%
    # validate - mode1
    agent = FullSystem()
    sys_data = {'PropagationFile': None, 'SOAFile': None,
                'CoeffFile': None,
                'load_checkpoint_dir': None,
                'PostWeightFile': None}
    agent.mode1_dmd(ref=True)
    agent.mode1_dmd(ref_point=551800)
    sys_data['PropagationFile'] = "**.pth.tar"
    agent.mode1_dmd(data_dict=sys_data, exp_flag=False, sys=True)
    
    agent.mode2_soa(ref=True)
    agent.mode2_soa(ref_point=551322, prev_flag=-1)
    sys_data['SOAFile'] = "**.mat"
    agent.mode2_soa(data_dict=sys_data, exp_flag=False, sys=True, prev_flag=-1)
    
    agent.mode3_cloop(ref=True)
    agent.mode3_cloop(ref_point=[34321, 35575], pow2=15.8, w_max=5.)
    
    # %%
    # validate - mode2
    agent = FullSystem()
    sys_data = {'PropagationFile': None, 'SOAFile': None,
                'CoeffFile': None,
                'load_checkpoint_dir': None,
                'PostWeightFile': None}

    agent.mode2_soa(ref=True)
    agent.mode2_soa(ref_point=551322, prev_flag=0)
    sys_data['SOAFile'] = "**.mat"
    agent.mode2_soa(data_dict=sys_data, exp_flag=False, sys=True, prev_flag=0)
    
    agent.mode3_cloop(ref=True)
    agent.mode3_cloop(ref_point=[34321, 35575], pow2=15.8, w_max=5.)
    
    # %%
    # validate - mode3
    agent = FullSystem()

    agent.mode5_test(ref_point1=551322, 
                     ref_point2= [582469, 582705, 582962, 583224, 583454], 
                     test_flag=False, prev_flag=0)
    agent.mode5_test(ref_point1=551322, 
                     ref_point2= [582469, 582705, 582962, 583224, 583454], 
                     test_flag=True, prev_flag=0)