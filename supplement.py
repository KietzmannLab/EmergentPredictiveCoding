#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:24:57 2022

@author: tempali

Code used to generate the supplemental figures
"""

# imports 

import torch
import numpy as np
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    DEVICE = 'cpu'


print('Using {}'.format(DEVICE))

R_PATH = 'Results/Supl/Data/'
F_PATH = 'Results/Supl/'
M_PATH = 'final_networks/mnist_nets/'
hdf_path = R_PATH+'network_stats.h5'

LOAD = False
SEED = 2553
if not os.path.isdir(os.path.dirname(R_PATH)):
    os.makedirs(os.path.dirname(R_PATH), exist_ok=True)
if not os.path.isdir(os.path.dirname(F_PATH)):
    os.makedirs(os.path.dirname(R_PATH), exist_ok=True)
    
if SEED != None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
# set up hdf5 file to store the results 
if not os.path.exists(hdf_path):
    store = pd.HDFStore(hdf_path)
    store.close()
INPUT_SIZE = 28*28
Z_CRIT = 2.576 #99%

# dataset loaders
import mnist

# framework files
import Network
import helper
import plot


# load data
train_set, validation_set, test_set = mnist.load(val_ratio=0.0)

# load pre and post MNIST networks
nets = [[], []]

n_instances = 10
# load networks for bootstrap
losses = ['l1_pre', 'l1_post']
# set up dictionaries to fill in the data
ec_results, ap_results, st_results = dict(), dict(), dict()

for loss_ind, loss in enumerate(losses):
    for i in range(0, n_instances):
        net = Network.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=INPUT_SIZE,
                title=M_PATH+"mnist_net_"+loss,
                device=DEVICE)
        net.load(i)
        nets[loss_ind].append(net)
net = nets[0][0]


      
# fig A1 & A2: plot digit predictions and median MNIST digit
digits = list(range(0, 10))
#------------------------------------------------------------------------------
fig, ax = plot.pred_after_timestep(net, test_set, mask=None, digits=digits, seed=2553)
plot.save_fig(fig, F_PATH+"A1", bbox_inches='tight')
#------------------------------------------------------------------------------
#fig A2: plot lesioned predictions + median MNIST digit
pred_mask = helper._pred_mask(net, test_set, train_set)
fig, ax = plot.pred_after_timestep(net, test_set, mask=pred_mask, digits=digits, seed=2553)
plot.save_fig(fig, F_PATH+"A2", bbox_inches='tight')
fig, ax = plot.display(train_set.x.median(dim=0).values, axes_visible=False)
plot.save_fig(fig, F_PATH+"A2_med", bbox_inches='tight')
#------------------------------------------------------------------------------
#fig A3: plot class specific lesioned predictions for each digit
masks = []
target = 7
c_mask = helper.pred_class_mask(net, test_set, target=target, Z_crit=Z_CRIT)
type_mask, type_stats = helper.compute_unit_types(net, test_set, train_set)
plot.topographic_distribution(type_mask.reshape(28,28))
fig, ax = plot.pred_after_timestep(net, test_set, mask=c_mask, digits=digits, seed=2553)
plot.save_fig(fig, F_PATH+"A3", bbox_inches='tight')
#------------------------------------------------------------------------------
# fig A4: topo dist untrained network 
untrained_net = Network.State(activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=INPUT_SIZE,
        hidden_size=INPUT_SIZE,
        title='',
        device=DEVICE)
type_mask, type_stats = helper.compute_unit_types(untrained_net, test_set, train_set)
fig = plot.topographic_distribution(type_mask.reshape(28, 28))
plot.save_fig(fig, F_PATH+"A4", bbox_inches='tight')