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
import seaborn as sns
import os 
import random
from matplotlib.ticker import MaxNLocator

def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()
plt.style.use('ggplot')

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    DEVICE = 'cpu'

print('Using {}'.format(DEVICE))

R_PATH = 'Results/Fig5/Data/'
F_PATH = 'Results/Fig5/'
M_PATH = 'final_networks/seeded_cifar_nets/'
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
if not os.path.exists(hdf_path) or LOAD==False:
    store = pd.HDFStore(hdf_path)
    store.close()  
INPUT_SIZE = 32*32*3


# dataset loaders
import cifar

import Network



train_set, validation_set, test_set = cifar.load(val_ratio=0.0, color=True)


Z_CRIT= 1.96 #95% CI
# cifar dimensions
nc, nx, ny = 3, 32, 32
nunits = INPUT_SIZE
n_instances = 10
nclasses = 10
seq_length = 10
LOSS_FN = 'l1_pre'
nets = [[], []]
c_types = ['cifar_net_'] # add cifar_latent_ if you want to test latent models
hid_size = [0, 32]
LOADS = [False, False]
for c, c_type in enumerate(c_types):
    for i in range(n_instances):
            net= Network.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=INPUT_SIZE+hid_size[c],
                title=M_PATH+c_type+str(i),
                device=DEVICE)
            
            net.load()
            nets[c].append(net)
        
batch_size=1
for c, c_type in enumerate(c_types):    
        #------------------------------------------------------------------------------
    ## Fig 5B: Topographic distribution and Input variance of prediction and error units 
    u_types = ['prediction', 'error', 'hybrid', 'all other units'] #['prediction', 'all other units']
    ## specify dictionary for all network instances
    pop_dict = {'Unit type':[], 'N': [], 'Median input variance':[], 'Network': []}
    for n, net in enumerate(nets[c]):
        net_path = R_PATH + 'net'+str(n)
        if not os.path.exists(hdf_path) or LOADS[0] == False:
            type_mask, type_stats = helper.compute_unit_types(net, test_set, train_set, seed=SEED)
            type_dict = {'Mask': type_mask, 'Stats': type_stats}
            typedf = pd.DataFrame(data=type_dict)
            # save dataframe 
            store = pd.HDFStore(hdf_path)
            store['type_stats_'+c_type+str(n)] = typedf
            store.close()
        else:
            store = pd.HDFStore(hdf_path)
            typedf = store['type_stats_'+c_type+str(n)]
            store.close()
            type_mask = typedf['Mask']
            type_stats = typedf['Stats']
        # reshape type mask for proper indexing
        type_mask = torch.tensor(list(type_mask))
        type_mask = type_mask.reshape(nunits)
        # # retrieve indices of unit types (prediction, error & hybrid)  
        #err_inds = [i for i, e in enumerate(type_mask) if e in [0,1]]
        pred_inds = [i for i, p in enumerate(type_mask) if p in [2,3]]
        #hybrid_inds = [i for i, h in enumerate(type_mask) if h in [4,5]]
        un_inds = [i for i, u in enumerate(type_mask) if u not in [2,3]]
     
        if not os.path.exists(hdf_path) or LOADS[1] == False:
            # # get prediction and error unit indices 
        
            # record input pixel variance per category
            var = torch.zeros(nclasses, INPUT_SIZE)
            # pred_inds, err_inds = [] , []
            for cat in range(nclasses):
                var[cat] = torch.var(test_set.x[test_set.indices[cat]],dim=0)
    
           
            # set up dictionary for single network
            var_dict = {'Unit type': [], 'Input variance': [], 'Nr classes':[], 'Categories': []}
        
            # pure prediction units
            for p in pred_inds:
                cpred, _, _ , _ = type_stats[p]
                var_pred = torch.zeros(len(cpred))
                for i, cat in enumerate(cpred):
                    targ_pred = (cat - 1) % seq_length
                    var_pred[i] = var[targ_pred, p]
                
                    
                var_dict['Unit type'].append('prediction')
                var_dict['Input variance'].append(var_pred.mean().item())
                var_dict['Nr classes'].append(len(cpred))
                var_dict['Categories'].append(cpred)
                
                
            # all other units
            for u in un_inds:
                var_u = torch.zeros(nclasses)
                for cat in range(nclasses):
                    var_u[cat] = var[cat, u]
                var_dict['Unit type'].append('all other units')
                var_dict['Input variance'].append(var_u.mean().item())
                var_dict['Nr classes'].append(0)
                var_dict['Categories'].append([])
            
            # create a dataframe to store the variances per unit type for single network
            netdf = pd.DataFrame(data=var_dict)
            # save dataframe 
            store = pd.HDFStore(hdf_path)
            store[c_type+str(n)] = netdf
            store.close()
            
        else: # load input variance data
            store = pd.HDFStore(hdf_path)
            netdf = store[c_type+str(n)]
            store.close()
            
            
        for u_type in u_types:
            pop_dict['Unit type'].append(u_type)
            if list(netdf.loc[netdf['Unit type'] == u_type]) == []: # unit type not in this network
                pop_dict['Median input variance'].append(0)
            elif u_type == 'hybrid': # take the prediction variance
                u_type_var = list(netdf.loc[netdf['Unit type'] == u_type]['Input variance'])
                pred_var = torch.tensor([p for p, e in u_type_var])
                pop_dict['Median input variance'].append(torch.median(pred_var).item())
            else:
                u_type_var = netdf.loc[netdf['Unit type'] == u_type]['Input variance'].median()
                pop_dict['Median input variance'].append(u_type_var)
            pop_dict['N'].append(len(netdf.loc[netdf['Unit type'] == u_type]))
            pop_dict['Network'].append('Network ' + str(n+1))
            
    popdf = pd.DataFrame(data=pop_dict)
    # save dataframe
    store = pd.HDFStore(hdf_path) 
    store['popinfo'] = popdf
   
    fig = plot.topographic_distribution(type_mask.reshape(3, 32, 32))
    plot.save_fig(fig, F_PATH + 'topographic_distribution_'+c_type)
    # plot input variance for each prediction and error unit
    fig, ax = plt.subplots(figsize=(7,7))

    df_prederr = popdf.loc[popdf['Unit type'].isin(['prediction', 'unspecified'])]
    
    ax = sns.barplot(x='Unit type', y='Median input variance', data=df_prederr, capsize=.2, color='#868484ff')
    plot.save_fig(fig, F_PATH + 'Input_variance_unit_types_'+c_type)
    
#------------------------------------------------------------------------------
    # ## fig 3A: compute average number of prediction and error units
    summary_stats = {'Unit type':[], 'Mean number of units':[], 'Std':[]}
    for u_type in u_types:
        mean = popdf.loc[popdf['Unit type'] == u_type]['N'].mean()
        std = popdf.loc[popdf['Unit type'] == u_type]['N'].std()
        summary_stats['Unit type'].append(u_type)
        summary_stats['Mean number of units'].append(mean)
        summary_stats['Std'].append(std)
    
    # Put stats in dataframe and save them to disk
    summary_stats = pd.DataFrame(data=summary_stats)
    store = pd.HDFStore(hdf_path) 
    store['summary_stats'+str(c_type)] = summary_stats
   
    print(summary_stats)


#------------------------------------------------------------------------------
    ## fig 5C: lesioning study CIFAR10
    # checkif samples are already computed
    if not os.path.exists(hdf_path) or LOADS[1] == False:
        bs_sample_dict = helper.bootstrap_model_activity(nets[0], train_set, test_set, seed=None, lesioned=True)
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
    # create figure plot mean values and 95% CI
    
    fig, (ax_top, ax_bott) = plt.subplots(2, 1, sharex=True)
    
       
    x = np.arange(1,seq_length+1)
    
    
    mu_norm = np.mean(norm_samples, axis=0) # empirical mean of original RNN   
    ax_bott.plot(x, mu_norm, label="original RNN", color= '#EE6666')
    lower_norm, upper_norm = helper.extract_lower_upper(bs_norm)
    
    ax_bott.fill_between(x, lower_norm, upper_norm, color='#EE6666', alpha=0.3) 
       
    
    mu_les = np.mean(lesion_samples, axis=0) # empirical mean of sample set
    ax_top.plot(x, mu_les, label="prediction units lesioned", color= '#EECC55')
    
    lower_les, upper_les = helper.extract_lower_upper(bs_lesion)
    ax_top.fill_between(x, lower_les, upper_les, color='#EECC55', alpha=0.3) 
    
    mu_cont = np.mean(cont_samples, axis=0) # empirical mean of sample set
    ax_bott.plot(x, mu_cont, label="control lesioning", color= '#5efc03')
    lower_cont, upper_cont = helper.extract_lower_upper(bs_cont)
    
    ax_bott.fill_between(x, lower_cont, upper_cont, color='#5efc03', alpha=0.3) 
    
    # set limits of axes using the bootstrap bounds
    ax_top.set_ylim(min(lower_les)-0.01, max(upper_les)+0.01) 
    ax_bott.set_ylim(min(lower_norm)-0.01, max(upper_norm)+0.01) 
    
    ax_bott.xaxis.set_major_locator(MaxNLocator(integer=True));
      
    ax_top.spines.bottom.set_visible(False)
    ax_bott.spines.top.set_visible(False)
    ax_top.spines.top.set_visible(False)
    
    ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_top.tick_params(bottom=False)
    
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_bott.get_legend_handles_labels()
    ax_bott.legend(h1+h2, l1+l2, loc=1, prop={'size': 8})
    d = .4  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bott.plot([0, 1], [1, 1], transform=ax_bott.transAxes, **kwargs)
    
    ax_top.grid(True); ax_bott.grid(True)
    ax_bott.tick_params(labeltop=False)  # don't put tick labels at the top
    
    plot.save_fig(fig, F_PATH + 'lesion_study_CIFAR_'+c_type)
store.close()
#------------------------------------------------------------------------------
# Uncomment if you want to look at latent models
# R_PATH_latent = 'Results/Fig5/Data/LatentModel/'
# latent_hdf_path = R_PATH_latent+'latent_network_stats.h5'
# if not os.path.isdir(os.path.dirname(R_PATH_latent)):
#     os.makedirs(os.path.dirname(R_PATH_latent), exist_ok=True)
# latent_store = pd.HDFStore(latent_hdf_path)
# N_latent = 32
# latent_nets = []
# for i in range(0, 10):
#     net32 = Network.State(activation_func=torch.nn.ReLU(),
#             optimizer=torch.optim.Adam,
#             lr=1e-4,
#             input_size=INPUT_SIZE,
#             hidden_size=INPUT_SIZE+N_latent,
#             title=M_PATH+c_types[1]+str(i),
#             device=DEVICE)
#     net32.load()
#     latent_nets.append(net32)
    
# latent_preds, non_latent_preds = [], []  
# for n, net in enumerate(latent_nets):
#     type_mask, type_stats = helper.compute_unit_types(net, test_set, train_set)
#     type_dict = {'Mask': type_mask, 'Stats': type_stats}
#     typedf = pd.DataFrame(data=type_dict)
#     # save dataframe 
#     latent_store = pd.HDFStore(latent_hdf_path)
#     latent_store['type_stats_'+c_types[1]+str(n)] = typedf
#     latent_store.close()
#     # reshape type mask for proper indexing
#     type_mask = type_mask.reshape(nunits+N_latent)
#     # # retrieve indices of unit types (prediction, error & hybrid)  
#     err_inds = [i for i, e in enumerate(type_mask) if e in [0,1]]
#     pred_inds = [i for i, p in enumerate(type_mask) if p in [2,3]]
#     hybrid_inds = [i for i, h in enumerate(type_mask) if h in [4,5]]
#     un_inds = [i for i, u in enumerate(type_mask) if u == 6]
#     for ind in pred_inds:
#         if ind > INPUT_SIZE:
#             latent_preds.append(ind)
#         else:
#             non_latent_preds.append(ind)
            
# print(latent_preds)
#------------------------------------------------------------------------------