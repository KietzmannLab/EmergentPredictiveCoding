import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import Dict

def get_time() -> int:
    '''Returns current time in ms'''
    return int(round(time.time() * 1000))

class Timer:
    def __init__(self):
        self.reset()

    def lap(self):
        self.t_lap = get_time()

    def get(self):
        return get_time() - self.t_lap

    def reset(self):
        self.t_total = get_time()
        self.t_lap = get_time()

    def __str__(self):
        t = self.get()
        ms = t % 1000
        t = int(t / 1000)
        s = t % 60
        t = int(t / 60)
        m = t % 60
        if t == 0:
            return "{}.{:03}".format(s,ms)
        else:
            t = int(t / 60)
            h = t
            if t == 0:
                return "{}:{:02}.{:03}".format(m,s,ms)
            else:
                return "{}:{:02}:{:02}.{:03}".format(h,m,s,ms)

def append_dict(dict_a:Dict[str,np.ndarray], dict_b:Dict[str,np.ndarray]):
    for k, v in dict_b.items():
        dict_a[k] = np.concatenate((dict_a[k], v))

def L1Loss(x:torch.FloatTensor):
   
    return torch.mean(torch.abs(x))

def L2Loss(x:torch.FloatTensor):
    
    return torch.mean(torch.pow(x, 2))

def Linear(x:torch.FloatTensor):
    return x

def parse_loss(args, terms):
    if args == None:
        return L1Loss, torch.tensor(0.0)
    
    pre, post, weights = terms
    arg1, arg2 = args.split('_')
    if arg1 == 'l1':
        loss_fn = L1Loss
    else:
        loss_fn = L2Loss
        
    if arg2 == 'pre':
        loss_arg = pre
    elif arg2 == 'post':
        loss_arg = post
    else:
        loss_arg = weights
        
    return loss_fn, loss_arg
            
def init_params(size_x, size_y):
    return ((torch.rand(size_x, size_y) * 2.) - 1. ) * np.sqrt(1. / size_x)

def normalize(x, p=2.0, dim=1):
    return F.normalize(x, p, dim)
