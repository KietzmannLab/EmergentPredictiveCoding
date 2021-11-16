import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import mnist

import cifar

from Dataset import Dataset
from ModelState import ModelState

import scipy.stats as st


#
#
#
# ----------        Script with helper functions for plot.py       -----------
#
#

#
# --- Helper functions for bootstrap plotting fig 2A, 4A & 5C ---
#
def compute_bootstrap(notn, meds, gmed, net, net_les=None, net_les_rev=None ,seq_length=10):
    """ compute bootstrap bounds for each time point"""  
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev = [],[],[],[], [], []
    
    for t in range(seq_length):
        bs_notn.append(bs.bootstrap(notn[:,t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_meds.append(bs.bootstrap(meds[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_gmed.append(bs.bootstrap(gmed[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_net.append(bs.bootstrap(net[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
       
        
        if net_les is not None:
            bs_netles.append(bs.bootstrap(net_les[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
            bs_netles_rev.append(bs.bootstrap(net_les_rev[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
    return bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev

def extract_lower_upper(bs_list):
    """
    wrapper function that extracts upper and lower bounds of the confidence
    interval 
    """
    lower, upper  = [bs.lower_bound for bs in bs_list], [bs.upper_bound for bs in bs_list]
    return lower,upper

#
# --- Helper function for Appendix A Figures A1 & A2 ---
#
def _run_seq_from_digit(digit, steps, net:ModelState, dataset:Dataset, mask=None):
    """Create sequences with the same starting digit through a model and return the hidden state

    Parameters:
        - digit: the last digit in the sequence
        - steps: sequence length, or steps before the sequence gets to the 'digit'
        - net: model
        - dataset: dataset to use
        - mask: mask can be used to turn off (i.e. lesion) certain units
    """
    fixed_starting_point = (digit - steps) % 10
    b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=True, fixed_starting_point=fixed_starting_point)
    batch = b.squeeze(0)

    h = net.model.init_state(1)
    for i in range(steps):
        h, l_a = net.model(batch[i], state=h)
        if mask is not None:
            h = h * mask

    return h.detach()

#
# --- Helper functions for lesion plots (Figures 4, 5C)
#
def _pred_mask(net:ModelState, test_set:Dataset, training_set:Dataset,  latent=False, reverse=False):
    """
    Wrapper function for calling the routine that computes the mask for the networks
    """
    pred_mask = _pred_mask_mad(net, test_set,training_set, latent=latent, reverse=reverse)
    return pred_mask

def _pred_mask_mad(net:ModelState, test_set:Dataset, training_set:Dataset, latent=False, reverse=False):
    """
    Returns a mask for the network units, where each entry is 1 if the 
    associated unit has a bias in its final time point median 
    preactivation and standard error in at least one class. 
    The rationale behind this approach is that a unit with nonzero preactivation
    has to have a functional role in supressing activity induced by the incoming
    digit since it would have been supressed by the objective function otherwise.

    """
    if type(training_set) is mnist.MNISTDataset:

        class_meds = mnist.medians(training_set)
    else: # cifar
        class_meds = cifar.medians(training_set)
        
    preact_stats = compute_preact_stats(net, test_set)

    med, mad = preact_stats[:, :, 0], preact_stats[:, :, 1]

    n_units, n_classes = net.model.W.shape[0], len(class_meds)
    A_mask = torch.zeros(n_units)


    for i in range(n_units):
        for j in range(n_classes):
             if (torch.abs(med[i][j]) - torch.abs(2.576*mad[i][j])) > 0:
                A_mask[i] = 1 # unit i is predictive for class j

    pred_mask = torch.ones(net.model.W.shape[0])

    if reverse:
        N_pred = sum(A_mask == 1).item()

        idx = (A_mask == 0).nonzero().flatten()
        perm = torch.randperm(len(idx))
        idx = idx[perm[:N_pred]]
        pred_mask[idx] = 0
    else:
        pred_mask[A_mask == 1] = 0
    return pred_mask
    

#
# --- Helper function (general) ---
#
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Adapted from https://stackoverflow.com/a/18926541
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
#
# --- Helper function for identifying prediction units 
# Figures 3A,B 4B, 5B,C, Appendix A3,A4 ---
#
def compute_preact_stats(net:ModelState, dataset:Dataset, nclasses=10, ntime=10):
    """
    Computer for each unit the average final time point median preactivation and MAD
    for each class
    
    Output: preact_stats matrix n_units x nclasses x 2
    """

    preact_stats = torch.zeros((net.model.hidden_size, nclasses, 2)) 
    
    # generate sequences that end in the same class
    for t in [ntime - 1]: # only look at final time point (0-indexed)
        for category in range(nclasses):
            starting_point = int(category - t + ntime)
            if starting_point > (ntime - 1): # cycle back
                starting_point -= ntime
            
            data, labels = dataset.create_batches(-1,ntime, shuffle=False,fixed_starting_point=starting_point)
            
            nb, ntime,batch_size,ninputs = data.shape
     
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            batch_size = data.shape[1]
            h_net = net.model.init_state(batch_size)
           
            for i in range(data.shape[0]): # calculate response variance of category up until t
                x = data[i]
                h_net, l_net = net.model(x, state=h_net)
                if i == t: 
                    med, mad= l_net[0].median(axis=0).values, torch.tensor(st.median_absolute_deviation(l_net[0].detach().numpy(), axis=0))
                  
            preact_stats[:, category, 0] = med
            preact_stats[:, category, 1] = mad
            
    return preact_stats.detach()  


# --- Helper functions that compute preactivation figures lesioned & non-lesioned 
# Figure 2A, 4B & 5C ---
#
def model_activity(net:ModelState,
                   training_set:Dataset,
                   test_set:Dataset,
                   seq_length=10,
                   data_type='mnist',
                   color=False,
                   save=True):
    """
    calculates model preactivation  and preactivation bounds 
    for unlesioned models 
    """
    nclass = 10 # change this if you want to change the number of classes
    # category medians and median for all images
    if data_type == 'mnist':
        meds = mnist.medians(training_set)
        global_median = training_set.x.median(dim=0).values
        N = 784
    elif data_type == 'cifar':
        meds = cifar.medians(training_set)
        global_median = training_set.x.median(dim=0).values
        N = 3072 # only compute results over non-latent units
    
    with torch.no_grad():
        data, labels = test_set.create_batches(-1, seq_length, shuffle=True)
        nb, ntime,batch_size,ninputs = data.shape
     

        data = data.squeeze(0)
        labels = labels.squeeze(0)
        batch_size = data.shape[1]

        # result lists
        mu_notn = []
        mu_meds = []
        mu_gmed = []
        mu_net = []
        mu_input = []
        mu_latent = []
       

        h_net = torch.zeros(batch_size, N)
        h_net = net.model.init_state(batch_size)
            
        for t in range(data.shape[0]):
        
            x = data[t]
            y = labels[t]

            # repeat global median for each input image
            gmedian = torch.zeros_like(x)
            gmedian[:,:] = global_median
            # find the corresponding median for each input image
            median = torch.zeros_like(x)
            for i in range(nclass):
                median[y==i,:] = meds[i]

            # calculate hidden state
            h_meds = (x - median)
            h_gmed = (x - gmedian)
           

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/net.model.input_size
            m_meds = h_meds.abs().sum(dim=1)/net.model.input_size
            m_gmed = h_gmed.abs().sum(dim=1)/net.model.input_size
            
                

            h_net, l_net = net.model(x, state=h_net) 
            m_net = torch.cat([a[:,:ninputs] for a in l_net], dim=1).abs().mean(dim=1).mean()
            m_input = torch.cat([a[:,:ninputs] for a in l_net], dim=1).abs().mean(dim=1).mean() 
            m_latent = torch.cat([a[:,ninputs:] for a in l_net], dim=1).abs().mean(dim=1).mean()
                
            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_net.append(m_net.mean().cpu().item())
            mu_input.append(m_input.mean().cpu().item())
            mu_latent.append(m_latent.mean().cpu().item())
            


        return data, np.array(mu_notn), np.array(mu_meds), np.array(mu_gmed), np.array(mu_net), np.array(mu_input), np.array(mu_latent)
            
def model_activity_lesioned(net:ModelState, training_set:Dataset, test_set:Dataset, seq_length=10, save=True, 
                            latent=False, data_type='mnist', reverse=False):
    """
    calculates model preactivation  and preactivation bounds 
    for lesioned models 
    """
    mask = _pred_mask(net, test_set, training_set= training_set, latent=latent, reverse=reverse)
    nclass = 10 # change this if you want to change the number of classes
    # meds: class-specific medians, global_median: median of the entire data set
    if data_type == 'mnist':
        meds = mnist.medians(training_set)
        global_median = training_set.x.median(dim=0).values

    elif data_type == 'cifar':
        meds = cifar.medians(training_set)
        global_median = training_set.x.median(dim=0).values 
  
    with torch.no_grad():
        data, labels = test_set.create_batches(-1, seq_length, shuffle=True)
        nb, ntime,batch_size,ninputs = data.shape
        data = data.squeeze(0)
        labels = labels.squeeze(0)
        batch_size = data.shape[1]
       
        # result lists
        mu_notn = []
        mu_meds = []
        mu_gmed = []
        mu_net = []
        mu_netles=[]
        mu_input = []
        mu_latent = []

        h_net = net.model.init_state(batch_size)
        h_netles = net.model.init_state(batch_size)
        
       
        for t in range(data.shape[0]):
            x = data[t]
            y = labels[t]
            
            # repeat global median for each input image
            gmedian = torch.zeros_like(x)
            gmedian[:,:] = global_median
            
            # find the corresponding median for each input image
            median = torch.zeros_like(x)
            for i in range(nclass):
                median[y==i,:] = meds[i]

            # calculate hidden state
            h_meds = (x - median)
            h_gmed = (x - gmedian)
            h_net, l_net = net.model(x, state=h_net)
            h_netles = h_netles * mask # perform lesion
            h_netles, l_netles = net.model(x, state=h_netles)

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/net.model.input_size
            m_meds = h_meds.abs().sum(dim=1)/net.model.input_size
            m_gmed = h_gmed.abs().sum(dim=1)/net.model.input_size
            m_net = torch.cat([a[:,:ninputs]for a in l_net], dim=1).abs().mean(dim=1) 
            m_netles = torch.cat([a[:,:ninputs] for a in l_netles], dim=1).abs().mean(dim=1)
            m_input = torch.cat([a[:,:ninputs] for a in l_netles], dim=1).abs().mean(dim=1).mean() 
            m_latent = torch.cat([a[:,ninputs:] for a in l_netles], dim=1).abs().mean(dim=1).mean()

            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_net.append(m_net.mean())
            mu_netles.append(m_netles.mean())
            mu_input.append(m_input.mean().cpu().item())
            mu_latent.append(m_latent.mean().cpu().item())
            
         
           
    return data, np.array(mu_notn), np.array(mu_meds), np.array(mu_gmed), np.array(mu_net), np.array(mu_netles), np.array(mu_input), np.array(mu_latent)
