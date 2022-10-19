import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import mnist

import cifar

from Dataset import Dataset
from ModelState import ModelState

import scipy.stats as st

# nested dict for color mapping unit types (fig 3A/fig5A)
CMAPPING = {'0': {'0': {'0':0, '1': 1}, '1': {'0':2,'1':3}},
            '1':{'0':4, '1':5}
            }
            
#
#
#
# ----------        Script with helper functions for plot.py       -----------
#
#



def _calc_energy(net, preactivation, energy, mask=None, med=False):
    if energy == 'ap':
        return _calc_ap(net, preactivation)
    elif energy =='st':
        return _calc_st(net, preactivation, mask, med)
    # return summary of ap st (Sengupta et al 2010)
    return (1/3)*_calc_ap(net, preactivation) + (2/3)*_calc_st(net, preactivation, mask, med)
           

def _calc_ap(net, preactivation):
    # calculate outputs
    act = F.relu(preactivation)
    return torch.abs(act)

def _calc_st(net, preactivation, mask, med=False):
    # calculate outputs
    act = F.relu(preactivation)
    if med:
        return act
    abs_W = net.model.W.detach()
    if mask is not None and len(mask.squeeze().shape) > 1: # weights need to be masked
        abs_W = abs_W * mask
    abs_act, abs_W = torch.abs(act), torch.abs(abs_W)
    synaptrans = torch.sum(abs_act.unsqueeze(-1) * abs_W, axis=1)

    return synaptrans
#
# --- Helper functions for bootstrap plotting fig 2A, 4A & 5C ---
#


def compute_pixel_variance(images):
    """
    computes variance of pixels for each channel seperately 
    """
    nsamples, nc, npix = images.shape
    pixel_var = torch.zeros(nc, npix)
    for c in range(nc):
        var_c = images[:, c, :].var(axis=0)
        pixel_var[c, :] = var_c
    return pixel_var

def find_pred_units(net, dataset, seq_length=10, Z_crit=2.576):
    nclasses=10
    preact_stats = compute_preact_stats(net, dataset)
    nunits = net.model.hidden_size
    pred_rule = torch.zeros(nunits, nclasses)
   
    for cls_plt in range(10):
        med, mad = preact_stats[:, cls_plt, 0], preact_stats[:, cls_plt, 1]
        
        # scale MAD to obtain a pseudo standard deviation
        # https://stats.stackexchange.com/questions/355943/scale-factor-for-mad-for-non-normal-distribution)
        
        for i in range(nunits): # Z_crit CI
            if (torch.abs(med[i]) - torch.abs(Z_crit*mad[i])) > 0:
                pred_rule[i, cls_plt] = 1
    return pred_rule

def find_error_units(net, test_set, seq_length=10, target=None, Z_crit=2.576):
    batch_size = 1

    class_error_units = torch.zeros(seq_length, net.model.hidden_size)
    t_ind = 8 # look at penultimate timepoint
    error_units = torch.zeros(net.model.hidden_size)
    for target in range(seq_length):
        starting_point = (target - t_ind) % 10
        # create normal sequences
        norm_seq = test_set.create_batches(batch_size, seq_length, fixed_starting_point=starting_point)
        # create distractor sequences 
        dis_seq = test_set.create_batches(batch_size, seq_length, distractor=True,fixed_starting_point=starting_point)
        # collect responses of networks on test set 
        responses = extract_responses(net, norm_seq)
        # collect responses of networks on distractor set
        d_responses = extract_responses(net, dis_seq)
        anomalies = detect_anomalies(net, responses, d_responses, Z_crit)
        for i in range(len(anomalies)):
            if anomalies[i] == 1:
                class_error_units[target, i] = 1 # 
                error_units[i] = 1
    return error_units, class_error_units

def detect_anomalies(net, responses, d_responses, Z_crit=2.576):
    t_ind = 8 # look at final time point
    n_units, n_samples = responses.shape[0], torch.tensor(responses.shape[1])
    mean_responses, std_responses = responses[:,:, t_ind].mean(axis=1), responses[:,:, t_ind].std(axis=1)
    mean_distractor,  std_d_responses = d_responses[:,:, t_ind].mean(axis=1), d_responses[:,:, t_ind].std(axis=1)
    
    #Z_crit = 2.576 #2.576 # 99%
    #Z_crit = 1.96
    anomalies = torch.zeros(n_units)
    
    for i in range(n_units):
        mu_i, mu_id = mean_responses[i], mean_distractor[i]
        # calculate standard errors and compute Z scores
        s_i, s_id = std_responses[i]/torch.sqrt(n_samples), std_d_responses[i]/torch.sqrt(n_samples)
        Z = torch.abs((mu_i - mu_id) / torch.sqrt(s_i**2 + s_id**2))
        if Z >= Z_crit: 
            anomalies[i] = 1
    return anomalies

def extract_responses(net, test_set, seq_length=10):
    """
    Collect responses from h_t from the network on the test
    data
    """
    test_data, test_labels = test_set

    n_units, nbatch, batch_size = net.model.W.shape[0], test_data.shape[0], 1
    responses = torch.zeros(n_units, nbatch, seq_length)
    
    for i,batch in enumerate(test_data):
        state = net.model.init_state(batch_size)
        
        for t in range(seq_length):
            #state = net.get_next_state(state,batch[t])
            state, l_terms=  net.model.forward(batch[t], state)
            a, h, W = l_terms
            # collect unit responses h_t 
            responses[:, i, t] = h.squeeze() #state.squeeze()
    return responses.detach()    


def compute_allclass_rd(net, test_set, seq_length=10):
    responses, drives =  list(zip(*[tuple(compute_responses_drive(net,test_set, target, seq_length))  for target in range(0, 9)]))  
    return torch.cat(responses, axis=1), torch.cat(drives, axis=1)

def compute_unit_types(net:ModelState, dataset:Dataset, training_set:Dataset=None, Z_crit=2.576, seed=2553):
    """
    Helper function that determines the types of units in the network
    
    The types are:
    0: pure error unit (e*)
    1: pure prediction unit (p*)
    2: hybrid (h*)
    3: unspecified (u)
    
    The pure units are devided based on how many classes they predict/error
    signal for:
        
    subtypes pure error units:
        0.1: 1 class (e1)
        0.2: multiclasses (e2)
        
    subtypes pure prediction units:
        1.1: 1 class (p1)
        1.2: multiclasses (p2)
        
    subtypes hybrid units:
        2.1: hybrid unit within (prediction and error unit for the same class) (h1)
        2.2: hybrid unit across (prediction and error unit for different classes) (h2)
        
    Resulting in 6 different typings
    
    These are assigned as:
        0-1: pure error (0: e1, 1: e2)
        2-3: pure prediction unit (2: p1, 3: p2)
        4-5: hybrid (4: h1, 5: h2)
        6: unspecified (u)
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    #preact_stats = compute_preact_stats(net, dataset)
    nunits, nclasses = net.model.hidden_size, 10
    pred_rule = find_pred_units(net, dataset, Z_crit=Z_crit)
    _, error_rule = find_error_units(net, dataset, Z_crit=Z_crit)
    units_stats = []
    
    for i in range(nunits):
        # count the number of classes the unit is predictive and error signaling for 
        #n_pred, n_err = 0,0
        # track the classes that the unit is predictive and error signaling for
        cpred, cerr = [], []
        for j in range(nclasses):
            if pred_rule[i,j] == 1:
                cpred.append(j)
                #n_pred += 1
            if error_rule[j,i] == 1:
                cerr.append(j)
                #n_err += 1
        # record if unit is predictive and error signaling
        within = list(set(cpred).intersection(cerr))
        unique_pred =  set(cpred).difference(set(cerr))
        unique_err = set(cerr).difference(set(cpred))
        if len(unique_pred) == 0 or len(unique_err) == 0: # cannot be across if one the lists are empty
            across = []
        else: # just take the union of the two sets
            across = list(unique_pred.union(unique_err))
        # record stats for unit i
        # cpred: the classes the unit is predictive for, cerr: the classes the unit is error signaling for
        # within: the classes the unit is both predictive and error signaling for
        # across: classes that the unit is either predictive or error signaling for
        units_stats.append((cpred, cerr, within, across))
        
    # parse  type (traverse decision tree)    
    # assign a type to the units
    units_types = torch.zeros(nunits)
    for i, stats in enumerate(units_stats):
        cpred, cerr, within, across = stats
        # decide if unspecified or not
        if len(cpred) == 0 and len(cerr) ==0: 
            units_types[i] = 6
            
        else:
            # decide if hybrid or not
            ply1 = int(len(within) > 0 or len(across) > 1)
           
            if ply1: #hybrid branche
                # decide if within/across
                ply2 = int((len(across) > 1))
                units_types[i] = CMAPPING[str(ply1)][str(ply2)]
                
            else: # PE branch 
                # decide if prediction unit
                ply2 = int((len(cpred) > 0))
                # decide if multiclass
                ply3 = int(len(cpred) > 1 or len(cerr) >1)
                units_types[i] = CMAPPING[str(ply1)][str(ply2)][str(ply3)]
            

    return units_types, units_stats

def compute_responses_drive(net, test_set, target=0, seq_length=10):
    """
    collect network responses & drive h_k  & network drive p_k+1
    look at h_k, p_k+1 (you want to correlate unit output i. vs. unit drive j.)
    
    Output: response matrix, drive matrix (NxOxK) where N=nr units, O= nr
    observations, K= sequence length = largest temporal history
    """
    batch_size = 1

    # record network predictions (activities of units)
    state = net.model.init_state(batch_size)
    
    # collect the response matrices in here
    response_list, drive_list, synaptrans_list = [], [], []
    # upper bound on temporal history since you need to be able to predict
    # one time step in the future and need to deal with 0-indexing
    K = seq_length-1 
    
    for k in range(K): 
        # determine where the sequence starts given temporal history k
        seq_start = (target - k) % 10 
        
        # create sequences
        batch_data, batch_labels = test_set.create_batches(batch_size, \
                                                           seq_length, fixed_starting_point=seq_start)
        nbatch = batch_data.shape[0]
        

        response_k, drive_k, synaptrans_k = torch.zeros((net.model.W.shape[0], nbatch)),\
            torch.zeros((net.model.W.shape[0], nbatch)), torch.zeros((net.model.W.shape[0], nbatch))
            

        abs_W = net.model.W.detach()#torch.abs(net.model.W.detach())
      
        # get observed responses  and predictions associated with target
        for i, batch in enumerate(batch_data):
                # move state forward to k
                for m in range(0, k+1): # m in [0,..,k]
                    state = net.get_next_state(state,batch[m])
                
                # collect drives and responses for target (h_k, p_k+1)
                response_k[:, i] = state
                drive_k[:, i] = net.predict(state).squeeze()
                synaptrans_k[:, i] = torch.sum(response_k[:, i].unsqueeze(-1) * abs_W, axis=1)
                # reset state 
                state = net.model.init_state(batch_size)
                
        # add the responses and drives to the list
        response_list.append(response_k)
        drive_list.append(drive_k)
        synaptrans_list.append(synaptrans_k)
   
    # construct full matrices, normalize and return them
    responses, drives, synaptrans = torch.stack(response_list, dim=-1), torch.stack(drive_list, dim=-1), torch.stack(synaptrans_list, dim=-1)

    return responses.detach(), drives.detach(), synaptrans.detach()


def compute_targ_pred_corrmat(responses, drive):
    """
    records the correlation between h_k and p_k for  temporal window
    k
    
    Output: a correlation matrix with dimensions N**2xK, where entry i,j contains the correlation
    between h^i_k and p^j_k+1, where K is the temporal window and N the
    number of units
    
 

    """
    n_units, n_obs, T = responses.shape
    K = T-1 # upper bound on temporal history
  
    corr_mat = torch.zeros((n_units, n_units, K))

    for k in range(0, K): 
        # compute correlation matrix for u_tk: (h_t-k, p_t)
        h_k, p_k1 = responses[:, :, k], drive[:, :, k+1]
        # compute correlation coefficient
        c_k = torch.tensor(np.ma.corrcoef(np.ma.masked_invalid(h_k), \
                                       np.ma.masked_invalid(p_k1)))[:n_units, n_units:] # only look at second quadrant
        for u_r in range(n_units):
            for u_d in range(n_units):
                corr_mat[u_r, u_d,k] = c_k[u_r, u_d]
    return corr_mat

def compute_post_drive_bootstrap(pred,error ,seq_length=9):
    """ compute bootstrap bounds for each time point"""  
    bs_pred, bs_error = [], []
    for t in range(seq_length):
        bs_pred.append(bs.bootstrap(pred[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_error.append(bs.bootstrap(error[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
    return bs_pred, bs_error

def compute_bootstrap(samples_list, seq_length=10):
    """ compute bootstrap bounds for each timepoint and set of samples"""
    bs_list = [[] for samples in samples_list]
    for t in range(seq_length):
        for i, bsamples in enumerate(bs_list):
            samples = samples_list[i]
            bsamples.append(bs.bootstrap(samples[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
    return bs_list

def compute_bootstrap_dep(notn, meds, gmed, net, net_les=None, net_les_rev=None ,seq_length=10):
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
    batch = b.squeeze() # removed 0 because of weird 

    h = net.model.init_state(1)
    for i in range(steps):
        # check if mask needs to be applied
        if mask is not None:
            # check if mask is for error or for prediction
            if len(mask.shape) > 1: # error mask
                h, l_a = net.model(batch[i], state=h, mask=mask)
            else:
                h, l_a = net.model(batch[i], state=h)
                h = h * mask
        else: 
            h, l_a = net.model(batch[i], state=h)
    return h.detach()

#
# --- Helper functions for lesion plots (Figures 4, 5C)
#

def pred_class_mask(net:ModelState, test_set:Dataset, target=0, Z_crit=2.576):
    """
    

    returns prediction unit mask for class: target

    """
    target = (target - 1) % 10 # activation will affect prediction one time step later
    n_units = net.model.W.shape[0]
    # shape: nunits x nclasses x 2
    preact_stats = compute_preact_stats(net, test_set)

    med, mad = preact_stats[:, :, 0], preact_stats[:,:, 1]
    
    pred_mask = torch.ones(n_units)
    for i in range(n_units):
        if (torch.abs(med[i][target]) - torch.abs(Z_crit*mad[i][target])) > 0:
            pred_mask[i] = 0 # unit i is predictive for class target
            
    return pred_mask
    
    
def _pred_mask(net:ModelState, test_set:Dataset, training_set:Dataset,  latent=False, reverse=False, Z_crit=2.576):
    """
    Wrapper function for calling the routine that computes the mask for the networks
    """
    pred_mask = _pred_mask_mad(net, test_set,training_set, latent=latent, reverse=reverse, Z_crit=Z_crit)
    return pred_mask

def _error_mask(net:ModelState, test_set, training_set, latent=False, reverse=False):
    """
    Knock out lateral connections between error units that are not
    prediction units
  
    """
    error_units, _ = find_error_units(net, test_set)
    error_indices = (error_units).nonzero().squeeze()
    pred_units = _pred_mask(net, test_set, training_set= training_set, latent=latent, reverse=reverse)
    pred_indices = (pred_units == 0).nonzero().squeeze()
    unique_error = error_indices[~error_indices.unsqueeze(1).eq(pred_indices).any(1)]
    unique_pred  = pred_indices[~pred_indices.unsqueeze(1).eq(error_indices).any(1)]
    mask = torch.ones(net.model.W.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i in error_indices and j in pred_indices:
                mask[i][j] = 0
            elif i in pred_indices and i in error_indices: # yellow unit
                mask[i][j] = 0 # prevent yellow units from inhibiting at t=1
    return mask
    
def _pred_mask_mad(net:ModelState, test_set:Dataset, training_set:Dataset, latent=False, reverse=False, Z_crit=2.576):
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
             if (torch.abs(med[i][j]) - torch.abs(Z_crit*mad[i][j])) > 0:
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
                #energy = _calc_energy(net, l_net[0], l_net[1])
                if i == t: 
                    med, mad= l_net[0].median(axis=0).values, torch.tensor(st.median_abs_deviation(l_net[0].detach().numpy(), axis=0, scale='normal'))
                  
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
    # calc energy demands for theoretical benchmarks 
    #meds = _calc_energy(net, meds, torch.nn.ReLU(meds)) # first preact
    #global_median = _calc_energy(net, global_median, torch.nn.ReLU(global_median))
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
            # calculate energy demands of x
            #x = _calc_energy(net, x, torch.nn.ReLU(x)) first test with preact
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
            # calculate energy demands for the network
            m_net = torch.cat([a[:,:ninputs] for a in l_net[0]], dim=1).abs().mean(dim=1).mean()
            m_input = torch.cat([a[:,:ninputs] for a in l_net[0]], dim=1).abs().mean(dim=1).mean() 
            m_latent = torch.cat([a[:,ninputs:] for a in l_net[0]], dim=1).abs().mean(dim=1).mean()
            
            # commented out for later analyses
            #m_net = _calc_energy(net, m_net, torch.nn.ReLU(m_net))
            #m_input = _calc_energy(net, m_input, torch.nn.ReLU(m_input))
            #m_latent = _calc_energy(net, m_latent, torch.nn.ReLU(m_latent))
                
            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_net.append(m_net.mean().cpu().item())
            mu_input.append(m_input.mean().cpu().item())
            mu_latent.append(m_latent.mean().cpu().item())
            


        return data, np.array(mu_notn), np.array(mu_meds), np.array(mu_gmed), np.array(mu_net), np.array(mu_input), np.array(mu_latent)


def bootstrap_model_activity(nets:[ModelState],
                   train_set:Dataset,
                   test_set:Dataset,
                   seq_length=10,
                   energy='ec',
                   lesioned=True,
                   lesion_type='pred',
                   latent=False,
                   seed=None,
                   Z_crit=2.576,
                   data_type='mnist'):
    """
    
    Calculates energy consumption of models and 
    all CI 99%/95% bootstrapped with replacement

    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    
     # initialize sample matrices 
    norm_samples = np.zeros((len(nets), seq_length))
    lesion_samples = np.zeros((len(nets), seq_length))
    cont_samples = np.zeros((len(nets), seq_length))
    
    for i, net in enumerate(nets):
        mu_norm, mu_les =\
        model_activity_lesioned(net, train_set, test_set, lesion_type='pred', seq_length=10, energy=energy, save=False,\
                                latent=False, data_type='mnist',Z_crit=Z_crit)
        
        # calculate energy curves with control lesion
        _, mu_cont=\
        model_activity_lesioned(net, train_set, test_set, lesion_type='pred', seq_length=10, energy=energy, save=False,\
                                latent=False, data_type='mnist', reverse=True, Z_crit=Z_crit)
        # fill sample matrices
        norm_samples[i, :] = mu_norm
        lesion_samples[i, :] = mu_les
        cont_samples[i, :] = mu_cont    
        
    # compute bootstrap bounds
    [bs_norm, bs_lesion, bs_cont] = compute_bootstrap([norm_samples, lesion_samples, cont_samples])
    # store samples and bs in dictionary
    bs_sample_dict = {'norm': [norm_samples], 'lesion': [lesion_samples], 'cont': \
                      [cont_samples], 'bs_norm': [bs_norm],'bs_lesion': [bs_lesion], 'bs_cont':[bs_cont]}
    return bs_sample_dict    
       
def model_activity_lesioned(net:ModelState, training_set:Dataset, test_set:Dataset, lesion_type='pred', 
                            seq_length=10, energy='ec', save=True, 
                            latent=False, data_type='mnist', reverse=False, Z_crit=2.576):
    """
    calculates model preactivation  and preactivation bounds 
    for lesioned models 
    """
    if data_type == 'mnist':
        batch_size = -1 # full dataset
    else:
        batch_size = 32
    if lesion_type == 'error':
        mask = _error_mask(net, test_set, training_set,latent=latent, reverse=False)
    else:
        mask = _pred_mask(net, test_set, training_set= training_set, latent=latent, reverse=reverse, Z_crit=Z_crit)
 
    with torch.no_grad():
        data, labels = test_set.create_batches(batch_size, seq_length, shuffle=True)
        nbatch, ntime, batch_size, ninputs = data.shape
        #data = data.squeeze(0)
        #labels = labels.squeeze(0)
        #batch_size = data.shape[1]
       
        # result lists
        mu_net, mu_netles = torch.zeros(ntime), torch.zeros(ntime)
        
        #mu_input = []
        #mu_latent = []
    
        h_net = net.model.init_state(batch_size)
        h_netles = net.model.init_state(batch_size)
        # create seperate states to prevent leaking across batches
        state = h_net.unsqueeze(0).repeat_interleave(nbatch, dim=0)
        lesioned_state = h_netles.unsqueeze(0).repeat_interleave(nbatch, dim=0)
        for t in range(ntime):
            m_net, m_netles = [], []
            for b in range(nbatch):
                h_net, h_netles = state[b], lesioned_state[b]
                x = data[b,t]
                h_net, l_net = net.model(x, state=h_net)
                if lesion_type == 'pred':
                    h_netles = h_netles * mask # perform lesion
                    h_netles, l_netles = net.model(x, state=h_netles)
                else: # lesion error units
                    h_netles, l_netles = net.model(x, state=h_net, mask=mask)
                
                # calculate energy of the hidden states
                if energy != 'pre':
                    l_net[0] = _calc_energy(net, l_net[0], energy)
                    l_netles[0] = _calc_energy(net, l_netles[0], energy, mask)
                
                #m_net[b,:] = torch.cat([a[:,:ninputs]for a in [l_net[0]]], dim=1).abs().mean(dim=1) 
             
                #m_netles[b,:] = torch.cat([a[:,:ninputs] for a in [l_netles[0]]], dim=1).abs().mean(dim=1)
                #m_input = torch.cat([a[:,:ninputs] for a in [l_netles[0]]], dim=1).abs().mean(dim=1).mean() 
                #m_latent = torch.cat([a[:,ninputs:] for a in [l_netles[0]]], dim=1).abs().mean(dim=1).mean()
                m_net += torch.cat([a[:,:ninputs] for a in [l_net[0]]], dim=1).abs().mean(dim=1).tolist()
                m_netles += torch.cat([a[:,:ninputs] for a in [l_netles[0]]], dim=1).abs().mean(dim=1).tolist()
                # update state and lesioned state for batch b
                state[b], lesioned_state[b] = h_net, h_netles
            # m_net = m_net.mean(axis=0)
            #m_netles = m_netles.mean(axis=0)
            # Calculate the mean
            mu_net[t], mu_netles[t] = torch.tensor(m_net).mean(), torch.tensor(m_netles).mean()
            #mu_net.append(m_net.flatten().mean())
            #mu_netles.append(m_netles.flatten().mean())
            #mu_input.append(m_input.mean().cpu().item())
            #mu_latent.append(m_latent.mean().cpu().item())   
    return np.array(mu_net), np.array(mu_netles)
