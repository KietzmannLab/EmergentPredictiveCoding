#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:24:57 2022

@author: tempali

In this analysis we compare how well L1 pre does vs. L1 post. 
"""

# imports 

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
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

R_PATH = 'Results/Fig2/Data/'
F_PATH = 'Results/Fig2/'
M_PATH = 'final_networks/seeded_mnist/'

hdf_path = R_PATH+'network_stats.h5'

LOAD = False
SEED = None
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
SEQ_LENGTH = 10
# dataset loaders
import mnist

# framework files
import Network
import helper
import plot
from matplotlib.ticker import MaxNLocator

# load data
train_set, validation_set, test_set = mnist.load(val_ratio=0.0)

# load pre, post MNIST networks
nets = [[], [], []]

n_instances = [list(range(0,10)), list(range(0,10)), [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
# load networks for bootstrap
losses = ['l1_pre','l1_post', [str(beta)+'beta'+'l1_postandl2_weights' for beta in [3708.0] ][0]]
# set up dictionaries to fill in the data
ec_results, ap_results, st_results, pre_results = dict(), dict(), dict(), dict()
result_list = [('ec', ec_results),('ap', ap_results), ('st', st_results), ('pre', pre_results)]
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
        

    
# # open file to read/writedata
store = pd.HDFStore(hdf_path)    
# fig 2A: RNN_pre performs better than RNN_post 
#------------------------------------------------------------------------------
if not os.path.exists(hdf_path) or LOAD == False:
    # calculate energy consumption for the losses 
    for loss_ind, loss in enumerate(losses):
        energies = dict() # dict of dicts
        for (ename, e_results) in result_list:
            bs_sample_dict = helper.bootstrap_model_activity(nets[loss_ind], train_set, test_set, energy = ename, seed=None, lesioned=False)
            en_samples = np.zeros((len(nets[loss_ind]), SEQ_LENGTH))
            for i, net in enumerate(nets[loss_ind]):
                mean_en, _ =\
                helper.model_activity_lesioned(net, train_set, test_set, lesion_type='pred', seq_length=10, energy=ename, save=False,\
                                        latent=False, data_type='mnist',Z_crit=Z_CRIT)
            
       
                # fill sample matrices  
                en_samples[i, :] = mean_en
                
            #compute bootstrap bounds and store results in dataframe
            [en_bounds] = helper.compute_bootstrap([en_samples])
            en_samples, en_bounds = bs_sample_dict['norm'], bs_sample_dict['bs_norm']
            energies[ename] = [en_samples[0], en_bounds[0]]
        df_loss = pd.DataFrame(data=energies)
        store[loss] = df_loss
    df_pre, df_post, df_pw = store['l1_pre'], store['l1_post'], store['3708.0betal1_postandl2_weights']
   
else:
    df_pre, df_post, df_pw = store['l1_pre'], store['l1_post'], store['3708.0betal1_postandl2_weights']
# retrieve energies and 
enames = list(zip(*result_list))[0]
for ename in enames:
    x = np.arange(1,SEQ_LENGTH+1)
    start_index = 0
    
    
    # get samples for pre post and weighted post
    pre_samples, pre_bootstraps  = df_pre[ename]
    act_samples, act_bootstraps = df_post[ename]
    pw_samples, pw_bootstraps = df_pw[ename]
    if ename == 'ap':
        fig, ax = plt.subplots(1,1)
        # add l1(preactivation) models    
        mu_pre = np.mean(pre_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_pre = ax.plot(x, mu_pre, label="RNN_pre", color= '#EE6666')
        lower_pre, upper_pre = helper.extract_lower_upper(pre_bootstraps)
        ax.fill_between(x, lower_pre[start_index:], upper_pre[start_index:], color='#EE6666', alpha=0.3) 
        
        mu_act = np.mean(act_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_post = ax.plot(x, mu_act, label="RNN_post", color= 'cornflowerblue')
        lower_act, upper_act = helper.extract_lower_upper(act_bootstraps)
        ax.fill_between(x, lower_act[start_index:], upper_act[start_index:], color='cornflowerblue', alpha=0.3) 
        #ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));
    elif ename == 'st':
        fig, (ax_top, ax_bott) = plt.subplots(2, 1, sharex=True)
       
   
    
        # add l1(preactivation) models    
        mu_pre = np.mean(pre_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_pre = ax_bott.plot(x, mu_pre, label="RNN_pre", color= '#EE6666')
        lower_pre, upper_pre = helper.extract_lower_upper(pre_bootstraps)
        ax_bott.fill_between(x, lower_pre[start_index:], upper_pre[start_index:], color='#EE6666', alpha=0.3) 
        
        mu_act = np.mean(act_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_post = ax_top.plot(x, mu_act, label="RNN_post", color= 'cornflowerblue')
        
        lower_act, upper_act = helper.extract_lower_upper(act_bootstraps)
        ax_top.fill_between(x, lower_act[start_index:], upper_act[start_index:], color='cornflowerblue', alpha=0.3) 
        # set limits of axes using the bootstrap bounds
        ax_top.set_ylim(min(lower_act)-0.015, max(upper_act)+0.015) 
        ax_bott.set_ylim(min(lower_pre)-0.015, max(upper_pre)+0.015) 
            
        ax_top.spines.bottom.set_visible(False)
        ax_bott.spines.top.set_visible(False)
        ax_top.spines.top.set_visible(False)
        #ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
        ax_top.tick_params(bottom=False)
        
        d = .4  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bott.plot([0, 1], [1, 1], transform=ax_bott.transAxes, **kwargs)
        
        
        
        
        h1, l1 = ax_top.get_legend_handles_labels()
        h2, l2 = ax_bott.get_legend_handles_labels()
        #ax_top.legend(h1+h2, l1+l2, loc=1, prop={'size': 8})
        
        
        
        ax_bott.xaxis.set_major_locator(MaxNLocator(integer=True));
        
        ax_top.grid(True)
        ax_bott.grid(True)
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
    else: # ename == total energy consumption
        fig, (ax_top, ax_bott) = plt.subplots(2, 1, sharex=True)
       
     
        # add l1(preactivation) models    
        mu_pre = np.mean(pre_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_pre = ax_bott.plot(x, mu_pre, label="RNN_pre", color= '#EE6666')
        lower_pre, upper_pre = helper.extract_lower_upper(pre_bootstraps)
        ax_bott.fill_between(x, lower_pre[start_index:], upper_pre[start_index:], color='#EE6666', alpha=0.3) 
        
        mu_act = np.mean(act_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        l1_post = ax_top.plot(x, mu_act, label="RNN_post", color= 'cornflowerblue')
        
        lower_act, upper_act = helper.extract_lower_upper(act_bootstraps)
        ax_top.fill_between(x, lower_act[start_index:], upper_act[start_index:], color='cornflowerblue', alpha=0.3) 
        
        mu_pw = np.mean(pw_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
        pw = ax_bott.plot(x, mu_pw, label="RNN_post+l2(W)", color= 'black')

        lower_pw, upper_pw = helper.extract_lower_upper(pw_bootstraps)
        ax_bott.fill_between(x, lower_pw[start_index:], upper_pw[start_index:], color='black', alpha=0.3) 
        # set limits of axes using the bootstrap bounds
        ax_top.set_ylim(min(lower_act)-0.015, max(upper_act)+0.015) 
        ax_bott.set_ylim(min(lower_pw)-0.015, max(upper_pw)+0.015) 
        
        
            
        ax_top.spines.bottom.set_visible(False)
        ax_bott.spines.top.set_visible(False)
        ax_top.spines.top.set_visible(False)
        #ax_top.xaxis.tick_top()
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
        ax_top.tick_params(bottom=False)
        
        d = .4  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
        ax_bott.plot([0, 1], [1, 1], transform=ax_bott.transAxes, **kwargs)
        
        
        
        
        h1, l1 = ax_top.get_legend_handles_labels()
        h2, l2 = ax_bott.get_legend_handles_labels()
        #ax_top.legend(h1+h2, l1+l2, loc=1, prop={'size': 8})
        
        
        
        ax_bott.xaxis.set_major_locator(MaxNLocator(integer=True));
        
        ax_top.grid(True)
        ax_bott.grid(True)
        ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
    plot.save_fig(fig, F_PATH + ename+'curve_MNIST')

#------------------------------------------------------------------------------
# fig 2C: Plot digit predictions 
trained_net = nets[0][0] # example trained network for visualisation
act_net = nets[1][0]
# example untrained network for visualisation
untrained_net = Network.State(activation_func=torch.nn.ReLU(),
          optimizer=torch.optim.Adam,
          lr=1e-4,
          input_size=INPUT_SIZE,
          hidden_size=INPUT_SIZE,
          title="",
          device=DEVICE)

# get visualisations for trained & untrained network 
X,P, _, T = plot.example_sequence_state(trained_net, test_set)
_,Pu, _, _ = plot.example_sequence_state(untrained_net, test_set)
_, Pcat, _, _ = plot.example_sequence_state(act_net, test_set)
# get visualisations 
fig, axes = plot.display(X, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"input_drive", bbox_inches='tight')

fig, axes = plot.display(P, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"internal_drive_trained", bbox_inches='tight')

fig, axes = plot.display(Pu, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"internal_drive_untrained", bbox_inches='tight')

fig, axes = plot.display(T, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"total_drive", bbox_inches='tight')
# get median total drive
M = mnist.medians(train_set)
fig, axes = plot.display(P + list(M), lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"total_drive_median_digit", bbox_inches='tight')

fig, axes = plot.display(Pcat, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight', colorbar=False)
plot.save_fig(fig, F_PATH+"internal_drive_l1postwithoutcolorb", bbox_inches='tight')
fig, axes = plot.display(Pcat, lims=None, shape=(10,1), figsize=(3,3), axes_visible=False, layout='tight', colorbar=True)
plot.save_fig(fig, F_PATH+"internal_drive_l1postw", bbox_inches='tight')
