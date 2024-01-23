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
from functions import get_device

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()
plt.style.use('ggplot')

DEVICE = get_device()

R_PATH = 'Results/Fig3/Data/'
F_PATH = 'Results/Fig3/'
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
    
INPUT_SIZE = 28*28


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


#------------------------------------------------------------------------------
## fig 3A: plot topographic distribution of unit types and pixel variance 
# use the first network for visualisation purposes
net = nets[0]
if not os.path.exists(hdf_path) or LOAD == False:
    type_mask, type_stats = helper.compute_unit_types(net, test_set, train_set)
    type_dict = {'Mask': type_mask, 'Stats': type_stats}
    typedf = pd.DataFrame(data=type_dict)
    # save dataframe 
    store = pd.HDFStore(hdf_path)
    store['type_stats'+str(net)] = typedf
    store.close()
else:
    store = pd.HDFStore(hdf_path)
    typedf = store['type_stats'+str(net)]
    store.close()
    type_mask = typedf['Mask']
    type_stats = typedf['Stats']
type_mask = type_mask.reshape(nc*nx,ny)
# plot topographic distribution and save figure 
fig = plot.topographic_distribution(type_mask)
plot.save_fig(fig, F_PATH + 'topographic_distribution_mnist')

#------------------------------------------------------------------------------
## Fig 3B: Input variance of prediction and error units 
u_types = ['prediction', 'error', 'hybrid', 'unspecified']
## specify dictionary for all network instances
pop_dict = {'Unit type':[], 'N': [], 'Median input variance':[], 'Network': []}
for n, net in enumerate(nets):
    net_path = R_PATH + 'net'+str(n)
    if not os.path.exists(hdf_path) or LOAD == False:
        type_mask, type_stats = helper.compute_unit_types(net, test_set, train_set)
        type_dict = {'Mask': type_mask, 'Stats': type_stats}
        typedf = pd.DataFrame(data=type_dict)
        # save dataframe 
        store = pd.HDFStore(hdf_path)
        store['type_stats_net'+str(n)] = typedf
        store.close()
    else:
        store = pd.HDFStore(hdf_path)
        typedf = store['type_stats_net'+str(n)]
        store.close()
        type_mask = typedf['Mask']
        type_stats = typedf['Stats']
    # reshape type mask for proper indexing
    type_mask = type_mask.reshape(nunits)
    # # retrieve indices of unit types (prediction, error & hybrid)  
    err_inds = [i for i, e in enumerate(type_mask) if e in [0,1]]
    pred_inds = [i for i, p in enumerate(type_mask) if p in [2,3]]
    hybrid_inds = [i for i, h in enumerate(type_mask) if h in [4,5]]
    un_inds = [i for i, u in enumerate(type_mask) if u == 6]
    
    if not os.path.exists(hdf_path) or LOAD == False:
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
            
        # pure error units
        for e in err_inds:
            _, cerr, _ , _ = type_stats[e]
            var_err = torch.zeros(len(cerr))
            for i, cat in enumerate(cerr):
                targ_err = cat
                var_err[i] = var[targ_err, e]
                
            var_dict['Unit type'].append('error')
            var_dict['Input variance'].append(var_err.mean().item())
            var_dict['Nr classes'].append(len(cerr))
            var_dict['Categories'].append(cerr)
        
        # hybrid units 
        for h in hybrid_inds:
            cpred, cerr, _ , _ = type_stats[h]
            var_pred, var_err = torch.zeros(len(cpred)), torch.zeros(len(cerr))
            for i, cat in enumerate(cpred):
                targ_pred = (cat - 1) % seq_length
                var_pred[i] = var[targ_pred, h]
                
            for i, cat in enumerate(cerr):
                targ_err = cat
                var_err[i] = var[targ_err, h]
                
            var_dict['Unit type'].append('hybrid')
            var_dict['Input variance'].append((var_pred.mean().item(), var_err.mean().item()))
            var_dict['Nr classes'].append((len(cpred), len(cerr)))
            var_dict['Categories'].append((cpred, cerr))
            
        # unspecified
        for u in un_inds:
            var_u = torch.zeros(nclasses)
            for cat in range(nclasses):
                var_u[cat] = var[cat, u]
            var_dict['Unit type'].append('unspecified')
            var_dict['Input variance'].append(var_u.mean().item())
            var_dict['Nr classes'].append(0)
            var_dict['Categories'].append([])
        
        # create a dataframe to store the variances per unit type for single network
        netdf = pd.DataFrame(data=var_dict)
        # save dataframe 
        store = pd.HDFStore(hdf_path)
        store['mnist_net'+str(net)] = netdf
        store.close()
    else: # load input variance data
        store = pd.HDFStore(hdf_path)
        netdf = store['mnist_net'+str(net)]
        store.close()
    for u_type in u_types:
        pop_dict['Unit type'].append(u_type)
        if u_type == 'hybrid':
            u_type_var = list(netdf.loc[netdf['Unit type'] == u_type]['Input variance'])
            pred_var, err_var = torch.tensor([p for p, e in u_type_var]), torch.tensor([e for p, e in u_type_var])
            # compute medians separately and add them to the df
            pop_dict['Median input variance'].append((torch.median(pred_var).item(), torch.median(err_var).item()))
        else:
            u_type_var = netdf.loc[netdf['Unit type'] == u_type]['Input variance'].median()
            pop_dict['Median input variance'].append(u_type_var)
        pop_dict['N'].append(len(netdf.loc[netdf['Unit type'] == u_type]))
        pop_dict['Network'].append('Network ' + str(n+1))
    
popdf = pd.DataFrame(data=pop_dict)
# save dataframe
store = pd.HDFStore(hdf_path) 
store['popinfo'] = popdf
store.close()
# plot input variance for each prediction and error unit
fig, ax = plt.subplots(figsize=(7,7))


df_prederr = popdf.loc[popdf['Unit type'].isin(['prediction', 'error'])]


ax = sns.barplot(x='Unit type', y='Median input variance', data=df_prederr, capsize=.2, color='#868484ff')

plot.save_fig(fig, F_PATH + 'Input_variance_unit_types_mnist')

#------------------------------------------------------------------------------
# ## fig 3: compute average number of prediction and error units
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
store['summary_stats'] = summary_stats
store.close()
print(summary_stats)
