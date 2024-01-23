import numpy as np
import torch
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cycler

from Dataset import Dataset
from ModelState import ModelState

# import helper functions for plotting
import helper
# Global matplotlib settings

colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                  '#EECC55', '#88BB44', '#FFBBBB'])


plt.rc('axes', axisbelow=True, prop_cycle=colors)
plt.rc('grid', linestyle='--')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('lines', linewidth=2)

# for a bit nicer font in plots
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.size'] = 18

plt.style.use('ggplot')

# ---------------     Helper/non core functions     ---------------
#

def save_fig(fig, filepath, bbox_inches=None):
    """Convenience wrapper for saving figures in a default "../Results/" directory and auto appends file extensions ".svg"
    and ".png"
    """
    fig.savefig(filepath + ".svg", bbox_inches=bbox_inches)
    fig.savefig(filepath + ".png", bbox_inches=bbox_inches)

def axes_iterator(axes):
    """Iterate over axes. Whether it is a single axis object, a list of axes, or a list of a list of axes
    """
    if isinstance(axes, np.ndarray):
        for ax in axes:
            yield from axes_iterator(ax)
    else:
        yield axes

def init_axes(len_x, figsize, shape=None, colorbar=False):
    """Convenience function for creating subplots with configuratons

    Parameters:
        - len_x: amount of subfigures
        - figsize: size per subplot. Actual figure size depends on the subfigure configuration and if colorbars are visible.
        - shape: subfigure configuration in rows and columns. If 'None', a configuration is chosen to minimise width and height. Default: None
        - colorbar: whether colorbars are going to be used. Used for figsize calculation
    """
    if shape is not None:
        assert isinstance(shape, tuple)
        ncols = shape[0]
        nrows = shape[1]
    else:
        nrows = int(np.sqrt(len_x))
        ncols = int(len_x / nrows)
        while not nrows*ncols == len_x:
            nrows -= 1
            ncols = int(len_x / nrows)

    #figsize = (figsize[1] * ncols + colorbar*0.5*figsize[0], figsize[0] * nrows)
    figsize = (figsize[1] * ncols + 0.5*colorbar*figsize[0], figsize[0] * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize)

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def display(imgs,
            lims=(-1.0, 1.0),
            cmap='seismic',
            size=None,
            figsize=(4,4),
            shape=None,
            colorbar=True,
            axes_visible=True,
            layout='regular',
            figax=None):
    """Convenience function for plotting multiple tensors as images.

    Function to quickly display multiple tensors as images in a grid.
    Image dimensions are expected to be square and are taken to be the square root of the tensor size.
    Tensor dimensions may be arbitrary.
    The images are automatically layed out in a compact grid, but this can be overridden.

    Parameters:
        - imgs: (list of) input tensor(s) (torch.Tensor or numpy.Array)
        - lims: pixel value interval. If 'None', it is set to the highest absolute value in both directions, positive and negative. Default: (-1,1)
        - cmap: color map. Default: 'seismic'
        - size: image width and height. If 'None', it is set to the first round square of the tensor size. Default: None
        - figsize: size per image. Actual figure size depends on the subfigure configuration and if colorbars are visible. Default: (4,4)
        - shape: subfigure configuration, in rows and columns of images. If 'None', a configuration is chosen to minimise width and height. Default: None
        - colorbar: show colorbar for only last row of axes. Default: False
        - axes_visible: show/hide axes. Default: True
        - layout: matplotlib layout. Default: 'regular'
        - figax: if not 'None', use existing figure and axes object. Default: None
        -cmaps: pass list of colormaps 
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        shape = (1,1)

    if size is not None:
        if not isinstance(size, tuple):
            size = (size, size)

    # convert to numpy if not already so
    imgs = [im.detach().cpu().numpy() if isinstance(im, torch.Tensor) else im for im in imgs]

    if lims is None:
        mx = max([max(im.max(),abs(im.min())) for im in imgs])
        lims = (-mx, mx)

    if figax is None:
        fig, axes = init_axes(len(imgs), figsize, shape=shape, colorbar=colorbar)
    else:
        fig, axes = figax
  
    for i, ax in enumerate(axes_iterator(axes)):
       
        img = imgs[i]
        ax.grid()
        if size is None:
            _size = int(np.sqrt(img.size))
            img = img[:_size*_size].reshape(_size,_size)
        else:
            img = img[:size[0]*size[1]].reshape(size[0],size[1])
  
        plot_im = ax.imshow(img,cmap=cmap)

        ax.label_outer()

        if lims is not None:
            plot_im.set_clim(lims[0], lims[1])

        if axes_visible == False:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  
    if colorbar:
  
        # if isinstance(axes, np.ndarray):
        #     for rax in axes:
        
        #         if isinstance(rax, np.ndarray):
        #             fig.colorbar(plot_im, ax=rax, shrink=0.80, location='right');
        #         else:
        #             fig.colorbar(plot_im, ax=rax, shrink=0.80);
        #else:
            #fig.colorbar(plot_im, ax=axes, shrink=0.80);
        if isinstance(axes, np.ndarray):
            fig.colorbar(plot_im, ax = axes[-1], location='right')
            fig.colorbar(plot_im, ax=axes, shrink=0.80);
            set_size(figsize[0]+0.5, figsize[1], axes[-1])
        else:
            fig.colorbar(plot_im, ax=axes)
            set_size(figsize[0]+0.5, figsize, axes[-1])

    if layout == 'tight':
        fig.tight_layout()

    return fig, axes



def scatter(x, y, discrete=False, figsize=(8,6), color = 'r', xlabel="", ylabel="", legend=None, figax=None):
    """Convenience function to create scatter plots

    Parameters:
        - x: x data points. Array or list of arrays.
        - y: y data points
        - discrete: whether xaxis ticks should be integer values. Default: False
        - figsize: matplotlib figsize. Default: (8,6)
        - xlabel: Default: ""
        - ylabel: Default: ""
        - legend: display legend. Default: None
        - figax: if not 'None', use existing figure and axes objects. Default: None
    """
    if figax is None:
        fig, axes = plt.subplots(1, figsize=figsize)
    else:
        fig, axes = figax

    if isinstance(x, list):
        ax = axes
        for i, _x in enumerate(x):
            ax.scatter(_x, y[i])
    else:
        axes.scatter(x, y, c=color)

    if discrete:
        axes.xaxis.set_major_locator(MaxNLocator(integer=True));
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid();

    if legend is not None:
        axes.legend(legend)

    return fig, axes

def training_progress(net:ModelState, save=True):
    """
    
    wrapper function that shows model training

    """
    fig, axes = init_axes(1, figsize=(6,8))


    axes.plot(np.arange(1, len(net.results["train loss"])+1), net.results["train loss"], label="Training set")
    axes.plot(np.arange(1, len(net.results["test loss"])+1), net.results["test loss"], label="Test set")

   
   
  
    axes.xaxis.set_major_locator(MaxNLocator(integer=True));
    axes.set_xlabel("Training time",fontsize=16)
    axes.set_ylabel("Loss",fontsize=16)
    axes.legend()
    axes.set_title('Loss network', fontsize=18)
    axes.grid(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    fig.tight_layout()

    if save is True:
        save_fig(fig, "training-progress", bbox_inches='tight')



#
# ---------------     Plotting code for figures paper     ---------------
#


#
# Figure 2A and Figure 4B, 5C
#

def bootstrap_post_dynamics(net_list:[ModelState],
                   test_set:Dataset,
                   seq_length=10):
    err_samples, pred_samples = np.zeros((len(net_list), seq_length-1)), np.zeros((len(net_list), seq_length-1))
    pred_stats, err_stats =  dict(), dict()
    for i, net in enumerate(net_list):
        responses, drives, synaptrans = helper.compute_responses_drive(net, \
                                                              test_set, target=0)  
            
    
        pred_units = helper.pred_class_mask(net, test_set, target=0)
        error_units, error_units_c = helper.find_error_units(net, test_set)
        pred_indices = (pred_units == 0).nonzero().squeeze()
        error_indices = (error_units_c[0,:]).nonzero().squeeze()
        # remove hybrid units
        error_indices = error_indices[~error_indices.unsqueeze(1).eq(pred_indices).any(1)]
        # select error & prediction units
        preds = synaptrans[pred_indices]
        error = synaptrans[error_indices]
        # get the average drives of the units 
        pred_curve = preds.mean(axis=1).mean(axis=0)
        error_curve = error.mean(axis=1).mean(axis=0)#.sort(dim=0).values[:len(pred_indices)]
        #error_curve =  top_error.mean(axis=0)
        
        
        # record curves
        pred_samples[i ,:] = pred_curve.cpu().numpy()
        err_samples[i ,:] = error_curve.cpu().numpy()
    pred_stats['samples'], err_stats['samples'] = pred_samples.mean(axis=0), err_samples.mean(axis=0)  
    bs_pred, bs_err = helper.compute_post_drive_bootstrap(pred_samples, err_samples)
    (h_pred, l_pred), (h_err, l_err) = helper.extract_lower_upper(bs_pred),  helper.extract_lower_upper(bs_err)
    pred_stats['h_bound'], pred_stats['l_bound'] = h_pred, l_pred
    err_stats['h_bound'], err_stats['l_bound'] = h_err, l_err
    return  pred_stats, err_stats






def display_activity_lossfn(model_results,
                           lesioned = False,
                           save=True,
                           reverse=False,
                           energy_type='ec',
                           data_type='mnist'): 
    """
    visualises energy consumption of networks trained with different 
    loss functions 
    """
    data, bootstraps, samples = model_results['l1_pre']
    # get results for activation
    _, _, _, bs_net_act, bs_netles_act, bs_netles_rev_act = model_results['l1_post'][1]
    _, _, _, net_act_samples, netles_act_samples, netles_act_rev_samples = model_results['l1_post'][-1]
    # # get results for activation + weights
    # _, _, _, bs_net_weight, bs_netles_weight, bs_netles_rev_weight = model_results['l1_postandl2_weights'][1]
    # _, _, _, net_weight_samples, netles_weight_samples, netles_weight_rev_samples = model_results['l1_postandl2_weights'][-1]
    
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev = bootstraps 
    notn_samples, meds_samples, gmed_samples, net_samples, net_les_samples, net_les_samples_rev = samples
    # create figure plot mean values and 95% CI
    if energy_type == 'ap':
        fig, (ax1) = plt.subplots(1, 1, sharex=True)
        ax2 = ax1
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.125)  # adjust space between axes
    start_index = 0
    
    if energy_type == 'ec':
        start_index = 1
        ax2.set_ylim(0.165, 0.20)  # RNN_pre
        ax1.set_ylim(0.7, 0.79) # RNN_post/RNN_post+weights 
    if energy_type == 'st':
        start_index = 0
        ax2.set_ylim(0.2, 0.26)  # RNN_pre
        ax1.set_ylim(0.5, 6) # RNN_post/RNN_post+weights 
        
    x = np.arange(start_index+1,data.shape[0]+1)    
    # add l1(preactivation) models    
    mu_net = np.mean(net_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
    l1_pre = ax2.plot(x, mu_net, label="RNN_pre", color= '#EE6666')
    lower_net, upper_net = helper.extract_lower_upper(bs_net)
    ax2.fill_between(x, lower_net[start_index:], upper_net[start_index:], color='#EE6666', alpha=0.3) 
    #ax2.tick_params(axis='y', labelcolor='#EE6666')
    # add l1(act) models  
    
    mu_net_act = np.mean(net_act_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
    l1_post = ax1.plot(x, mu_net_act, label="RNN_post", color= 'cornflowerblue')
    #ax1.tick_params(axis='y', labelcolor='m')
    lower_net_act, upper_net_act = helper.extract_lower_upper(bs_net_act)
    ax1.fill_between(x, lower_net_act[start_index:], upper_net_act[start_index:], color='cornflowerblue', alpha=0.3) 
    
    # add l1(post) + l2(weights) models    
    # mu_net_weight = np.mean(net_weight_samples, axis=0)[start_index:] # empirical mean of reservoir activity   
    # l1l2_postW = ax1.plot(x, mu_net_weight, linestyle='--', label="RNN_post+weights", color= 'cyan')
    # lower_net_weight, upper_net_weight = helper.extract_lower_upper(bs_net_weight)
    # ax2.fill_between(x, lower_net_weight[start_index:], upper_net_weight[start_index:], color='cyan', alpha=0.3) 
   
   
    
    
    if energy_type == 'ec' or energy_type=='st':
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.spines.top.set_visible(False)
        #ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax1.tick_params(bottom=False)

        d = .4  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


    
    if energy_type == 'ec' or energy_type=='st':
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=0)
    else:
        ax1.legend()
    
    
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True));

    ax1.grid(True)
    ax2.grid(True)
    #ax1.spines['right'].set_visible(False)
    #ax1.spines['top'].set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    #ax2.xaxis.tick_bottom()
    #ax1.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=8)
    #ax1.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=8)
    #ax1.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=8)
    #ax2.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=8)
    

    if save is True:
        if lesioned:
            save_fig(fig, "energy_curves" + "_"+ energy_type + "_"+data_type+"/lesioned-model-activity", bbox_inches='tight')
        else:
            save_fig(fig, "energy_curves" + "_"+ energy_type + "_"+data_type+"/model-activity", bbox_inches='tight')
    return fig, (ax1, ax2)

def display_model_activity(model_results,
                           lesioned = False,
                           save=True,
                           reverse=False,
                           data_type='mnist'): 
    """
    visualises energy consumption of networks trained with different 
    loss functions 
    """
    data, bootstraps, samples = model_results['l1_pre']
    # get results for activation
    _, _, _, bs_net_act, bs_netles_act, bs_netles_rev_act = model_results['l1_post'][1]
    _, _, _, net_act_samples, netles_act_samples, netles_act_rev_samples = model_results['l1_post'][-1]
    # get results for activation + weights
    _, _, _, bs_net_weight, bs_netles_weight, bs_netles_rev_weight = model_results['l1_postandl2_weights'][1]
    _, _, _, net_weight_samples, netles_weight_samples, netles_weight_rev_samples = model_results['l1_postandl2_weights'][-1]
    
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev = bootstraps 
    notn_samples, meds_samples, gmed_samples, net_samples, net_les_samples, net_les_samples_rev = samples
    # create figure plot mean values and 95% CI
    fig, axes = plt.subplots(1, figsize=(14,10))
   
    x = np.arange(2,data.shape[0]+1)
    #mu_gmed = np.mean(gmed_samples, axis=0) # empirical mean of global median
    # axes.plot(x, mu_gmed, label="dataset median inhibition", color= '0.2')
    # lower_gmed, upper_gmed = helper.extract_lower_upper(bs_gmed)
    # axes.fill_between(x, lower_gmed, upper_gmed, color='0.2', alpha=0.3) 
    # mu_meds = np.mean(meds_samples, axis=0) # empirical mean of category median
    # axes.plot(x, mu_meds, label="category median inhibition", color='0.7')
    # lower_med, upper_med = helper.extract_lower_upper(bs_meds)
    # axes.fill_between(x, lower_med, upper_med, color='0.7', alpha=0.3) 
    
    # add a space between bounds and data that is input-dependent
    #axes.plot([],[], linestyle='', label=' ')
    
    #if not lesioned: # add input drive to the figure
        #mu_notn = np.mean(notn_samples, axis=0) # empirical mean of input drive sequences
        #axes.plot(x, mu_notn, label="input", color= '#3388BB')
        #lower_notn, upper_notn = helper.extract_lower_upper(bs_notn)
        #axes.fill_between(x, lower_notn, upper_notn, color='#3388BB', alpha=0.3) 
    # add l1(preactivation) models    
    mu_net = np.mean(net_samples, axis=0)[1:] # empirical mean of reservoir activity   
    l1_pre = axes.plot(x, mu_net, label="RNN_pre", color= '#EE6666')
    lower_net, upper_net = helper.extract_lower_upper(bs_net)
    axes.fill_between(x, lower_net[1:], upper_net[1:], color='#EE6666', alpha=0.3) 
    axes.tick_params(axis='y', labelcolor='#EE6666')
    # add l1(act) models  
    axes2= axes.twinx()
    mu_net_act = np.mean(net_act_samples, axis=0)[1:] # empirical mean of reservoir activity   
    l1_post = axes2.plot(x, mu_net_act, label="RNN_post", color= 'm')
    axes2.tick_params(axis='y', labelcolor='m')
   
    lower_net_act, upper_net_act = helper.extract_lower_upper(bs_net_act)
    axes2.fill_between(x, lower_net_act[1:], upper_net_act[1:], color='m', alpha=0.3) 
    
    # add l1(post) + l2(weights) models    
    mu_net_weight = np.mean(net_weight_samples, axis=0)[1:] # empirical mean of reservoir activity   
    l1l2_postW = axes2.plot(x, mu_net_weight, linestyle='--', label="RNN_post+weights", color= 'm')
    lower_net_weight, upper_net_weight = helper.extract_lower_upper(bs_net_weight)
    #axes2.fill_between(x, lower_net_weight[1:], upper_net_weight[1:], color='m', alpha=0.1) 
   
    
   
    if lesioned: # add lesioned reservoir to the figure 
        mu_netles = np.mean(net_les_samples, axis=0) # empirical mean of sample set
        axes.plot(x, mu_netles, label="prediction units lesioned", color= '#EECC55')
        lower_netles, upper_netles = helper.extract_lower_upper(bs_netles)
        axes.fill_between(x, lower_netles, upper_netles, color='#EECC55', alpha=0.3) 
        if reverse:
            mu_netles_rev = np.mean(net_les_samples_rev, axis=0) # empirical mean of sample set
            axes.plot(x, mu_netles_rev, linestyle='--', label="error units lesioned", color= '#5efc03')
            lower_netles_rev, upper_netles_rev = helper.extract_lower_upper(bs_netles_rev)
        
            axes.fill_between(x, lower_netles_rev, upper_netles_rev, color='#5efc03', alpha=0.3) 

    axes.xaxis.set_major_locator(MaxNLocator(integer=True));
    
    axes.legend(fontsize=18,labelspacing=0.1, facecolor='0.95')
    
    # lns = [l1_pre, l1_post, l1l2_postW]
    # labs = [l.get_label() for l in lns]
    # axes.legend(lns, labs, loc=0)
    h1, l1 = axes.get_legend_handles_labels()
    h2, l2 = axes2.get_legend_handles_labels()
    axes.legend(h1+h2, l1+l2, loc=0)
    axes.grid(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)
    axes.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)
    axes2.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)
    axes2.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)


    if save is True:
        if lesioned:
            save_fig(fig, "preactivation_curves" + data_type+"/lesioned-model-activity", bbox_inches='tight')
        else:
            save_fig(fig,  "preactivation_curves" + data_type+"/model-activity", bbox_inches='tight')
    return fig, axes,axes2 

#
# Figure 2C 
#
def example_sequence_state(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=False):
    """
    visualises input and internal drive for a sample sequence
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    batches, _ = dataset.create_batches(batch_size=-1, sequence_length=10, shuffle=False)

    ex_seq = batches[0,:,:,:]
    input_size = ex_seq.shape[-1] # make sure we only visualize input units and no latent resources
    X = []; P = []; H=[]; T=[]; L=[]
    
    h = net.model.init_state(ex_seq.shape[1])
   
    for x in ex_seq:

        p = net.predict(h, latent)
        h, l_a = net.model(x, state=h)
        #x_mu, p_mu = x[:,:input_size].mean(dim=0), p[:,:input_size].mean(dim=0)
        #x_std, p_std = x[:, :input_size].std(dim=0), p[:, :input_size].std(dim=0)
        X.append(x[0,:input_size].detach().cpu())
        P.append(p[0,:input_size].detach().cpu())
        H.append(h[0,:input_size].detach().cpu())
        T.append(l_a[0][0,:input_size].detach().cpu())
        # standardize input and internal drive so that they're on the same scale
        # x_scaled = (x[0,:input_size]- x_mu) / x_std
        # p_scaled = (p[0,:input_size]- p_mu) / p_std
        # x_scaled[torch.isnan(x_scaled)]=0 
        # p_scaled[torch.isnan(p_scaled)]=0 
        #t  = x_scaled.detach().cpu() + p_scaled.detach().cpu()
        
        if latent: # look at latent unit drive 
            L.append(p[:,input_size:].mean(dim=0).detach().cpu())

    # fig = plt.figure(figsize=(3,3))
    # if latent:
    #     fig, axes = display(X+P+L,  shape=(10,3), figsize=(3,3), axes_visible=False, layout='tight')
    # else:
    #      fig, axes = display(X+P+H+T,  shape=(10,3), figsize=(3,3), axes_visible=False, layout='tight')

    
    # if save is True:
    #     save_fig(fig, "example_sequence_state", bbox_inches='tight')
    return X, P, H, T

#
# Figure 3B/5B
#
def pixel_variance(pix_var):
    """Plot variance of each pixel and channel"""
    vmi, vma = 0, pix_var.max()
    fig, ax  = plt.subplots(1, 1)
    im = ax.imshow(pix_var, vmin=vmi, vmax=vma, cmap='gray')
    ax.grid(False)
    fig.colorbar(im)
    return fig
        
def  topographic_distribution(type_mask):
    """
    
    plots topographic distribution of prediction and error units in
    data space.
    
    """
    
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    blues = ["#3399ff", "#0000ff"] # pure error
    reds = ["#ff9999","#ff0000"] # pure prediction
    browns = ["#ffff00ff","#ffaa00ff"] # hybrid
    grey= ["#cccccc"] # unspecified
    combined = blues + reds + browns + grey
    cmap = ListedColormap(sns.color_palette(combined).as_hex())
    
    if len(type_mask.shape) > 2: # channel dim exists
        fig,ax  = plt.subplots(1, 3)
        nc, nx, ny = type_mask.shape
        for c in range(nc):
            ax[c].imshow(type_mask[c], cmap=cmap)
            ax[c].grid(False)
    else:
        fig,ax  = plt.subplots(1, 1)
        ax.imshow(type_mask, cmap=cmap)
        ax.grid(False)
    
    return fig


        

#
# Appendix A Figures A1 & A2 & A3
#

def pred_after_timestep(net, dataset, mask=None, digits=[0], seed=2553):
    """
    visualises internal drive after 0-9 preceding frames.
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    
        
    imgs= []
    ntime=10
    nunits = net.model.input_size


    for digit in digits:
        imgs = imgs + [net.predict(torch.zeros(1,nunits))] +\
            [net.predict(helper._run_seq_from_digit(digit, i, net, dataset, mask=mask)).mean(dim=0) for i in range(1,ntime)]
   
    fig, axes = display(imgs, shape=(ntime, len(digits)), axes_visible=False)


    fig.tight_layout()
    return fig, axes

def pred_after_timestep_predonly(net, dataset, mask, pred_mask, digits=[0], seed=2553):
    """
    visualises internal drive after 0-9 preceding frames. Where the internal
    drive only comes from prediction units.
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    
        
    imgs= []
    ntime=10
    nunits = net.model.input_size


    for digit in digits:
        imgs = imgs + [net.predict(torch.zeros(1,nunits))] +\
            [net.predict_predonly(helper._run_seq_from_digit(digit, i, net, dataset, mask=mask), pred_mask=pred_mask).mean(dim=0) for i in range(1,ntime)]
   
    fig, axes = display(imgs, shape=(ntime, len(digits)), axes_visible=False)


    fig.tight_layout()
    return fig, axes     

    
def color_code_pred_units(mnist, cifar, save=False):
    mnist_net, test_set_m = mnist
    cifar_net, test_set_c = cifar
    pred_mnist, pred_cifar = torch.zeros(mnist_net.model.W.shape[0]), torch.zeros(cifar_net.model.W.shape[0])
    for target in range(0, 10):
        pred_mask_m = helper.pred_class_mask(mnist_net, test_set_m, target=target)
        pred_mask_c = helper.pred_class_mask(cifar_net, test_set_c, target=target)
        pred_mnist += pred_mask_m
        pred_cifar += pred_mask_c
        
    fig, axes  = plt.subplots(1, 2)
    cmap_base = 'seismic'
    vmin, vmax = 0, 1
    cmap = helper.truncate_colormap(cmap_base, vmin, vmax)
    im1, im2 = axes[0].imshow(pred_mnist.view(28,28),  cmap=cmap),  axes[1].imshow(pred_cifar.view(32*32,3)[...,0],  cmap=cmap)
    axes[0].grid(False); axes[1].grid(False)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    fig.colorbar(im2, cax=cbar_ax)

   

    if save is True:
        save_fig(fig, "pixel_var_mnist_cifar", bbox_inches='tight')