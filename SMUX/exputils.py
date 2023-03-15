from numpy.lib.function_base import re
import pyvisa
from pyvisa.constants import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import struct
from time import sleep
import socket
import serial
import os
import scipy.io as scio
import math
from ALP4 import *
import cv2

class AWG:
    wait_time = 0.05
    def __init__(self, rm, address, ch=[1, 2]):
        self.awg = rm.open_resource(address, VI_EXCLUSIVE_LOCK)
        
        self.awg.write_termination = '\n'
        self.awg.read_termination = '\n'
        if isinstance(ch, list):
            for c in ch:
                self.awg.write(f':OUTP{c} ON')
                self.awg.write(f':OUTP{c} OFF')
                self.awg.write(f':SOUR{c}:BURS OFF')
                # self.awg.write(f':OUTP{c}:LOAD 50')
        else:
            self.awg.write(f':OUTP{ch} ON')
            self.awg.write(f':OUTP{ch} OFF')
            self.awg.write(f':SOUR{ch}:BURS OFF')
            # self.awg.write(f':OUTP{ch}:LOAD 50')
    
    
    def write_wait(self, order):
        if isinstance(order, list):
            for o in order:
                self.awg.write(o)
                sleep(self.wait_time)
        else:
            self.awg.write(order)
            sleep(self.wait_time)
        return 
    
    
    def ramp(self, ch=1, freq=1e3, amp=5, offset=0, sym=50):
        self.write_wait([f':SOUR{ch}:APPL:RAMP {freq},{amp},{offset}',
                         f':SOUR{ch}:FUNC:RAMP:SYMM {sym}'])
    
    
    def arbitrary(self, data01, block_len=6000, ch=1, sample_rate=1607600, amp=5, offset=2.5, wait_time=0.0001):
        self.write_wait(f':SOUR{ch}:APPL:ARB {sample_rate},{amp},{offset}')
        len_d = data01.shape[0]
        prefix = f':SOUR{ch}:DATA:DAC16 VOLATILE,'
        d_l = []
        block_num = int(np.ceil(len_d / block_len))
        for i in range(block_num):
            tmp = data01[(i * block_len):(i * block_len + block_len)]
            tmp = np.round(tmp * 0x3fff).astype(np.int_)            
            if i == block_num - 1:
                d_str = prefix + 'END,#'
            else:
                d_str = prefix + 'CON,#'
            bytes_len = f'{2 * tmp.shape[0]}'
            d_str += f'{len(bytes_len)}{bytes_len},'
            d_str = d_str.encode() + bytes.fromhex(''.join([f'{_:04x}' for _ in tmp])) # + '\n'.encode()
            d_l.append(d_str)
        for d_str in d_l:
            self.awg.write_raw(d_str)
            sleep(wait_time)
            
            
    def burst(self, ch=1, nc=2,
              mode='TRIG', # TRIG INF GAT
              slope='POS', # POS NEG
              source='INT', # EXT MAN INT
              period=10e-3,
              trigo='POS', # POS NEG OFF
              idle='FPT', # FPT TOP CENTER BOTTOM
              delay=0
              ):
        self.write_wait([f':SOUR{ch}:BURS ON',                  f':SOUR{ch}:BURS:MODE {mode}',
                        f':SOUR{ch}:BURS:NCYC {nc}',            f':SOUR{ch}:BURS:TRIG:SLOP {slope}',
                        f':SOUR{ch}:BURS:TRIG:SOUR {source}',   f':SOUR{ch}:BURS:INT:PER {period:f}',
                        f':SOUR{ch}:BURS:TRIG:TRIGO {trigo}',   f':SOUR{ch}:BURS:IDLE {idle}'])
        self.write_wait([f':SOUR{ch}:BURS:TDEL {delay}'])
        
    def square(self, ch=1, freq=1000, amp=4, offset=2, duty=50):
        self.write_wait(f':SOUR{ch}:APPL:SQU {freq},{amp},{offset},0')
        self.write_wait(f':SOUR{ch}:FUNC:SQU:DCYC {duty}')
        
    def dc(self, ch=1, offset=2):
        self.write_wait(f':SOUR{ch}:APPL:DC 1,1,{offset}')
    
    def set_state(self, state=[1, 1], ch=[1, 2], burst=False):
        if isinstance(ch, list):
            for c, s in zip(ch, state):
                ss = 'ON' if s == 1 else 'OFF'
                self.write_wait(f':OUTP{c} {ss}')
                if burst:
                    self.write_wait(f':SOUR{c}:BURS {ss}')
        else:
            ss = 'ON' if state == 1 else 'OFF'
            self.write_wait(f':OUTP{ch} {ss}')
            if burst:
                self.write_wait(f':SOUR{ch}:BURS {ss}')
    
            
    def reset(self, ch=[1, 2]):
        if isinstance(ch, list):
            if 1 in ch:
                self.write_wait([':SOUR1:BURS OFF',  ':OUTP1 OFF'])
            if 2 in ch:
                self.write_wait([':SOUR2:BURS OFF',  ':OUTP2 OFF'])
        else:
            if ch == 1:
                self.write_wait([':SOUR1:BURS OFF',  ':OUTP1 OFF'])
            elif ch == 2:
                self.write_wait([':SOUR2:BURS OFF',  ':OUTP2 OFF'])
    
    
    def exp_reshape(self, raw_data):
        def re(r):
            # 2, fm_10, b, od, oh*ow
            r = r.reshape((*r.shape[:3], -1, 5, r.shape[-1]))
            # 2, fm_10, b, od_5, 5, oh*ow
            r = np.transpose(r, [2, 5, 0, 1, 3, 4])
            r_s = r.shape
            # b, oh*ow, 2, fm_10, od_5, 5
            r = np.reshape(r, (-1, 5))
            # b*oh*ow*2*fm_10*od_5, 5
            return r, r_s
            # loop_num, 5
        if isinstance(raw_data, tuple):
            res = []
            for rr in raw_data:
                r, raw_shape = re(rr)
                res.append(r)
            return (*res, raw_shape)
        else:
            res, raw_shape = re(raw_data)
            return res, raw_shape
 

def fmt(s):
    if s == 0 or s == 1 or s == -1:
        return str(int(s))
    else:
        return f'{s:.4f}'
        
class AWG4(AWG):
    def __init__(self, rm, address, ch=[1, 2]):
        super().__init__(rm, address, ch)
    
    def arbitrary(self, data01, block_len=6000, ch=1, sample_rate=1607600, amp=5, offset=2.5, wait_time=0.0001):
        self.write_wait(f':SOUR{ch}:APPL:USER {sample_rate/data01.shape[0]},{amp},{offset}')
        data01 = data01 * 2 - 1
        d_str = f':SOUR{ch}:DATA VOLATILE,' + ','.join([fmt(_) for _ in data01])
        sleep(wait_time)
            
    def set_amp_off(self, ch=1, amp=1, offset=1./2, sample_rate=1607600):
        self.write_wait([f':SOUR{ch}:APPL:USER {sample_rate/5000},{amp},{offset}'])
    
class AWG7(AWG):
    def __init__(self, rm, address, ch=[1, 2], awg_max=[1, 1]):
        super().__init__(rm, address, ch)
        for c in ch:
            self.write_wait([f':OUTPut{c}:PATH DCAmp'])
        self.wait_time = 0.1
        for c, a_max in zip(ch, awg_max):
            self.set_amp_off(ch=c, amp=a_max, offset=a_max/2)
        for c in ch:
            self.burst(ch=c, trig_src='I')
        self.awg.timeout = 100000
            
    def arbitrary(self, data01, wfm_name, ch=1, sample_rate=1607600, wait_time=0.0001):
        data11 = data01 * 2 - 1
        data11 = np.round(data11 * 0x7fff).astype(np.int_)
        self.write_wait([f':SOURce{ch}:CASSet:CLEar',
                         f':SOURce:FREQuency:CW:FIXed {sample_rate}'])
        if wfm_name not in self.awg.query(':WLISt:LIST?'):
            self.write_wait(f':WLISt:WAVeform:NEW {wfm_name},31000,REAL')
        # tmp = np.round(np.abs(data11) * 0x7fff).astype(np.int_)
        d_str = f':WLISt:WAVeform:DATA {wfm_name},1,{data11.shape[0]},#'
        bytes_len = f'{2 * data11.shape[0]}'
        d_str += f'{len(bytes_len)}{bytes_len},'
        # d_str = d_str.encode() + bytes.fromhex(''.join([f'{_:04x}' for _ in tmp]))
        d_str = d_str.encode() + struct.pack('>' + 'h'*data11.shape[0], *data11)
        self.awg.write_raw(d_str)
        sleep(wait_time)
    
    def burst(self, ch=1, trig_src='A', interval=0.001):
        orders = [f':SOURce{ch}:RMODe TRIGgered',
                  f':SOURce{ch}:TINPut {trig_src}TRigger',
                  f':TRIGger:LEVel 1.1,{trig_src}TRigger',
                  f':OUTPut{ch}:WVALue:ANALog:STATe FIRSt']
        if trig_src == 'I':
            orders[2] = f':TRIGger:INTerval {interval}'
        self.write_wait(orders)

    def sync(self, ch=[1, 2]):
        d = ':SOURce:SYNC'
        for c in ch:
            d  += f'CH{c},'
        self.write_wait(d[:-1])
    
    def set_amp_off(self, ch=1, amp=1, offset=1./2):
        self.write_wait([f':SOURce{ch}:VOLTage:LEVel:IMMediate:AMPLitude {amp}',
                         f':SOURce{ch}:VOLTage:LEVel:IMMediate:OFFSet {offset}'])
    
    def set_state(self, state=[1, 1, 1], ch=[1, 3, 4], burst=False, sync=False):
        orders = []
        if sync:
            orders.append(f':AWGControl:CHANnel{ch[0]}:RUN')
        for c, s in zip(ch, state):
            if ~sync:
                orders.append(f':AWGControl:CHANnel{c}:RUN')
            orders.append(f':OUTPut{c}:STATe {s}')
        self.write_wait(orders)
        
        
class OSCP:
    def __init__(self, rm, address="USB0::0x0699::0x052C::C033930::INSTR"):
        self.oscp = rm.open_resource(address, VI_EXCLUSIVE_LOCK, VI_NULL)
        # self.oscp.timeout = 100000
        # self.mdo.read_termination = '\n'
        # self.mdo.write_termination = '\n'

    def read_waveform(self, ch=2, return_x=False, start_stop=None):
        oscp = self.oscp
        oscp.set_visa_attribute(VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS)
        oscp.set_visa_attribute(VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS)
        oscp.write("header off")
        
        oscp.write(f"DATa:SOUrce CH{ch}")
        elements = self.query_reco()
        if start_stop is None:
            oscp.write(f'DATa:STARt 0')
            oscp.write(f'DATa:STOP {elements - 1}')
        else:
            oscp.write(f'DATa:STARt {start_stop[0]}')
            oscp.write(f'DATa:STOP {start_stop[1]}')
        yoffset = oscp.query_ascii_values("WFMOutpre:YOFF?", 'f')[0]
        ymult = oscp.query_ascii_values("WFMOutpre:YMULT?", 'f')[0]
        yzero = oscp.query_ascii_values('WFMOutpre:YZEro?')[0]
        oscp.write("DATA:ENCDG RIBINARY;WIDTH 1")
        oscp.write("CURVE?")
        oscp.flush(VI_WRITE_BUF|VI_READ_BUF_DISCARD)
        oscp.set_visa_attribute(VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE)
        c = oscp.read_bytes(1)
        assert(c==b'#')
        c = oscp.read_bytes(1)
        assert(b'0' <= c <= b'9')
        count = int(c) - int(b'0')
        c = oscp.read_bytes(count)
        elements = int(c)
        c = oscp.read_bytes(elements)
        oscp.flush(VI_WRITE_BUF | VI_READ_BUF_DISCARD)
        res = np.array(struct.unpack('b' * elements, c))
        res = (res - yoffset) * ymult + yzero
        if return_x:
            return np.r_[:elements] * self.query_xspec(), res
        else:
            return res
        

    def query_xspec(self):
        return self.oscp.query_ascii_values('WFMOutpre:XINcr?')[0]

    def query_reco(self):
        return self.oscp.query_ascii_values("hor:reco?", 'd')[0]

    def set_xscale(self, scale):
        self.oscp.write(f'HORIZONTAL:SCALE {scale:.2e}')

    def set_yscale(self, ch, scale):
        self.oscp.write(f'CH{ch}:Scale {scale:.2e}')
        
    def set_recordlength(self, length):
        self.oscp.write(f'HORizontal:RECOrdlength {length}')

class Laser:
    def __init__(self, host='cobrite.local', port=2000):
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.connect((self.host, self.port))

    def sendScpi(self, command):
        # print("TX: " + command)
        self.socket.sendall(bytearray(command + "\n", 'utf-8'))
        reply = ""
        while reply.find('\n') < 0:
            reply = reply + self.socket.recv(1024).decode("utf-8")
        # print("RX: " + reply)
        return reply

    def send_wait(self, command):
        self.sendScpi(command)
        self.sendScpi('*WAI')
    
    def set_wav(self, wav, port=1):
        self.send_wait(f'WAV 1,1,{port},{wav}')
        
    def set_power(self, power, port=1):
        self.send_wait(f'POW 1,1,{port},{power}')
        
    def set_state(self, state, port=1, power=None, wav=None):
        self.send_wait(f'STAT 1,1,{port},{state}')
        if power is not None:
            self.send_wait(f'POW 1,1,{port},{power}')
        if wav is not None:
            self.send_wait(f'WAV 1,1,{port},{wav}')

    def close(self):
        self.reset()
        self.socket.close()
    
    def reset(self):
        self.set_state(state=0, port=1)
        self.set_state(state=0, port=2)


class BiasControl:
    def __init__(self, port='COM5'):
        self.bc = serial.Serial(port=port, baudrate=57600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, 
                    stopbits=serial.STOPBITS_ONE, timeout=5)
        assert(self.bc.isOpen())
    
    def query_hex(self, hex_ord):
        self.bc.write(bytes.fromhex(hex_ord))
        self.bc.flush()
        sleep(.1)
        return self.bc.read(9)
    
    def get(self, mode='bias'):
        if mode == 'bias':
            r = self.query_hex('68 00 00 00 00 00 00')
        elif mode == 'vpi':
            r = self.query_hex('69 00 00 00 00 00 00')
        elif mode == 'optical':
            r = self.query_hex('67 00 00 00 00 00 00')
        return struct.unpack('<f', r[1:5])
    
    def set_bias(self, v):
        v = int(v * 1000)
        sgn = 0 if v >= 0 else 1
        r = self.query_hex(f"6C00{abs(v):04x}{sgn:02d}0000")
        assert(r[1] == 0x11)        
        
    def set_point(self, point='NULL'):
        if point == 'Quad':
            r = self.query_hex('6A 02 00 00 00 00 00')
        else:
            r = self.query_hex('6A 01 00 00 00 00 00')        
        assert(r[1] == 0x11)
    
    def set_mode(self, mode='auto'):
        if mode == 'man':
            r = self.query_hex('6B 02 00 00 00 00 00')
        else:
            r = self.query_hex('6B 01 00 00 00 00 00')
        assert(r[1] == 0x11)
        
    def auto_man(self, wait_time=5):
        self.set_mode('auto')
        self.set_point('NULL')
        sleep(wait_time)
        self.set_mode('man')
    
    def close(self):
        self.bc.close()
        

class DMD():
    def __init__(self, nbImg, ill_time=33278, pic_time=33334):
        self.ill_time = ill_time
        self.pic_time = pic_time
        self.DMD = ALP4(version='4.3')
        self.DMD.Initialize()
        self.border_tblr = [420, 119, 940, 439]
        self.nbImg = nbImg
        self.SeqId = None
        
    
    def put_imgs(self, imgs, transform=True):
        self.reset()
        self.SeqId = self.DMD.SeqAlloc(nbImg=self.nbImg, bitDepth=1)
        if transform:
            imgSeq = np.concatenate([self.transform(_) for _ in imgs])
        else:
            imgSeq = np.concatenate([_.ravel() for _ in imgs])
        self.DMD.SeqPut(imgData=imgSeq, SequenceId=self.SeqId, PicLoad=self.nbImg)
        # self.DMD.SetTiming(pictureTime=self.pic_time, illuminationTime=self.ill_time)
        self.DMD.SetTiming(illuminationTime=65, triggerInDelay=1)
        self.DMD.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_SLAVE)
        self.DMD.ProjControl(controlType=ALP_PROJ_STEP, value=ALP_EDGE_FALLING)
        self.DMD.Run(SequenceId=self.SeqId)
    
    def put_img(self, img, transform=True):
        self.reset()
        self.SeqId = self.DMD.SeqAlloc(1, bitDepth=1)
        if transform:
            imgSeq = self.transform(img)
        else:
            imgSeq = img.ravel()
        self.DMD.SeqPut(imgData=imgSeq, SequenceId=self.SeqId)
        self.DMD.SetTiming(pictureTime=self.pic_time, illuminationTime=self.ill_time)
        self.DMD.ProjControl(controlType=ALP_PROJ_MODE, value=ALP_MASTER)
        # self.DMD.ProjControl(controlType=ALP_PROJ_STEP, value=ALP_EDGE_FALLING)
        self.DMD.Run(SequenceId=self.SeqId)
    
    def transform(self, img):
        img = img.astype(np.uint8)
        img = cv2.resize(img, [541, 541], interpolation=cv2.INTER_NEAREST)
        img = cv2.copyMakeBorder(img, *self.border_tblr, borderType=cv2.BORDER_CONSTANT, value=0)
        return img.ravel()
    
    def set_position(self, rx=None, ry=None):
        if rx is not None and ry is not None:
            h, w = 1080, 1920
            mh, mw = h//2 + ry, w//2 + rx
            hw1 = 541 // 2
            self.border_tblr = [mh - hw1, h - (mh + hw1) - 1,
                                mw - hw1, w - (mw + hw1) - 1]
        print(self.border_tblr)
        tmp = np.ones((541, 541)).astype(np.uint8) * 255
        tmp = cv2.copyMakeBorder(tmp, *self.border_tblr, borderType=cv2.BORDER_CONSTANT, value=0)
        # cv2.imwrite('tmp.png', tmp)
        return tmp
    
    def put_circle(self, center, radius):
        self.reset()
        img = np.zeros((1080, 1920), dtype=np.uint8)
        cv2.circle(img, center[0], int(radius[0]), (255, 255), thickness=-1)
        cv2.circle(img, center[1], int(radius[1]), (255, 255), thickness=-1)
        self.put_img(img, transform=False)

    
    def put_white(self):
        img = np.ones((1080, 1920), dtype=np.uint8) * 255
        self.put_img(img, transform=False)
    
    
    def reset(self):
        self.DMD.Halt()
        if self.SeqId is not None:
            self.DMD.FreeSeq(self.SeqId)
            self.SeqId = None
        
        
    def close(self):
        self.reset()
        self.DMD.Free()


class BaseAgent():
    def __init__(self):
        print('Connect to Device...')
        self.rm = pyvisa.ResourceManager()
        self.oscp = OSCP(self.rm, "USB0::0x0699::0x052C::C033930::INSTR")
        self.awg1 = AWG(self.rm, "USB::0x1AB1::0x0642::DG1ZA234205592::INSTR", [1, 2])
        self.awg2 = AWG(self.rm, "USB::0x1AB1::0x0642::DG1ZA230400233::INSTR", 1)
        self.laser = Laser()
        # self.bc = BiasControl("COM5")
        self.bc1 = BiasControl("COM4")

    def close(self):
        self.laser.close()
        # self.bc.close()
        self.bc1.close()
        self.awg1.reset()
        self.awg2.reset()


if __name__ == '__main__':
    laser = BiasControl()
    laser.set_wav(1550.00)
    laser.set_state(1, 1)
    laser.set_wav(1551.00, 2)
    pass