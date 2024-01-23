#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:08:50 2022

@author: tempali
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

import helper
import plot
import os 
from functions import get_device

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()
plt.style.use('ggplot')

DEVICE = get_device()

R_PATH = 'Results/Fig4/Data/'
F_PATH = 'Results/Fig4/'
M_PATH = 'patterns_rev/seeded_mnist/'
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

import Network



train_set, validation_set, test_set = mnist.load(val_ratio=0.0)
# mnist dimensions
nc, nx, ny = 1, 28, 28
nunits = nx*ny
n_instances = 10
seq_length = 10
nclasses = 10
LOSS_FN = 'l1_pre'
nets = []

for i in range(n_instances):
        net= Network.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            input_size=INPUT_SIZE,
            hidden_size=INPUT_SIZE,
            title=M_PATH+"mnist_net_"+LOSS_FN,
            device=DEVICE)
        
        net.load(i)
        nets.append(net)
        
batch_size=1
# Use the first network for visualisation purposes
net = nets[0]
store = pd.HDFStore(hdf_path)
#------------------------------------------------------------------------------
## fig 4A: Lesion study MNIST
if not os.path.exists(hdf_path) or LOAD == False:
   bs_sample_dict = helper.bootstrap_model_activity(nets, train_set, test_set, seed=None, lesioned=True)
   
   
   les_df = pd.DataFrame(data=bs_sample_dict)
   
   store['lesionstudy'] = les_df

else:
    les_df = store['lesionstudy']
    norm_samples, lesion_samples, cont_samples = store['norm'][0], \
        store['lesion'][0],  store['cont'][0]
    [bs_norm, bs_lesion, bs_cont] = les_df['bs_bounds']
 
# get samples
norm_samples, lesion_samples, cont_samples= les_df['norm'][0], les_df['lesion'][0], les_df['cont'][0]
# get bs_bounds
bs_norm, bs_lesion, bs_cont = les_df['bs_norm'][0], les_df['bs_lesion'][0],  les_df['bs_cont'][0]
# plot results
fig, ax = plt.subplots(1, 1)

   
x = np.arange(1,seq_length+1)
ax.set_xticks(x)

mu_norm = np.mean(norm_samples, axis=0) # empirical mean of original RNN   
ax.plot(x, mu_norm, label="original RNN", color= '#EE6666')
lower_norm, upper_norm = helper.extract_lower_upper(bs_norm)

ax.fill_between(x, lower_norm, upper_norm, color='#EE6666', alpha=0.3) 
   

mu_les = np.mean(lesion_samples, axis=0) # empirical mean of sample set
ax.plot(x, mu_les, label="prediction units lesioned", color= '#EECC55')

lower_les, upper_les = helper.extract_lower_upper(bs_lesion)
ax.fill_between(x, lower_les, upper_les, color='#EECC55', alpha=0.3) 

mu_cont = np.mean(cont_samples, axis=0) # empirical mean of sample set
ax.plot(x, mu_cont, label="control lesioning", color= '#5efc03')
lower_cont, upper_cont = helper.extract_lower_upper(bs_cont)

ax.fill_between(x, lower_cont, upper_cont, color='#5efc03', alpha=0.3) 

ax.legend()

plot.save_fig(fig, F_PATH + 'lesion_study_MNIST')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
## fig 4B: Visualise internal network drive for lesioned and non-lesioned network
## only visualise [0] 
pred_mask = helper._pred_mask(net, test_set, train_set)
# show internal network drive normal network
fig_norm, _ = plot.pred_after_timestep(net, test_set)
plot.save_fig(fig_norm, F_PATH + 'Internal drive without lesions')
# show lesioned internal network drive network
fig_les, _ = plot.pred_after_timestep(net, test_set, mask=pred_mask)
plot.save_fig(fig_les, F_PATH + 'Internal drive with lesions')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
## fig 4C: Compute and plot postsynaptic drive dynamics

if not os.path.exists(hdf_path) or LOAD == False:
    pred_stats, err_stats = plot.bootstrap_post_dynamics(nets, test_set)
    dyn_dict = {'pred': pred_stats, 'err':err_stats}
    postdyn_df = pd.DataFrame(dyn_dict)
   
    store['postdyn'] = postdyn_df
    
else:
    postdyn_df = store['postdyn']
    pred_stats, err_stats =  postdyn_df['pred'], postdyn_df['err']  


fig, (ax1, ax2) = plt.subplots(2, 1)
   
ax2.set_ylim(-11, 2.5)  # pred stats
ax1.set_ylim(-0.06, -0.01) # err stats
#fig.subplots_adjust(hspace=0.1)  # adjust space between axes
x = np.arange(0,9,1)
# plot the lines and confidence bounds 
ax1.plot(x, err_stats['samples'], color='b', label='error units ')
ax1.fill_between(x, err_stats['l_bound'], err_stats['h_bound'], color='b', alpha=0.3) 


ax2.plot(x, pred_stats['samples'],  color='r', label='prediction units')
ax2.fill_between(x, pred_stats['l_bound'], pred_stats['h_bound'], color='r', alpha=0.3)

ax1.legend()

ax2.legend()

plt.gca().set_aspect('auto')
plt.grid(True)

fig.tight_layout()
plot.save_fig(fig, F_PATH + 'postsynaptic_drive_dynamics')
store.close()
#------------------------------------------------------------------------------