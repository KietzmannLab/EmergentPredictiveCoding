import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cycler
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import mnist
from functions import *
from Dataset import Dataset
from ModelState import ModelState


# Global matplotlib settings

colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                  '#EECC55', '#88BB44', '#FFBBBB'])


plt.rc('axes', axisbelow=True, prop_cycle=colors)
plt.rc('grid', linestyle='--')
plt.rc('xtick', direction='out', color='black')
plt.rc('ytick', direction='out', color='black')
plt.rc('lines', linewidth=2)

#plt.rc('errorbar', capsize=4)
# for a bit nicer font in plots
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.size'] = 18
mpl.use('ps')
plt.style.use('ggplot')

# ---------------     Convenience functions     ---------------
#

def save_fig(fig, name, bbox_inches=None):
    """Convenience wrapper for saving figures in a default "../figures/" directory and auto appends file extension ".svg"
    """
    filepath = "Figures/" + name
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath + ".svg", bbox_inches=bbox_inches)

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

    figsize = (figsize[1] * ncols + colorbar*0.5*figsize[0], figsize[0] * nrows)

    return plt.subplots(nrows, ncols, figsize=figsize)

def display(imgs,
            lims=(-1.0, 1.0),
            cmap='seismic',
            size=None,
            figsize=(4,4),
            shape=None,
            colorbar=False,
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
        - colorbar: show colorbar for each row of axes. Default: False
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
  
        if isinstance(axes, np.ndarray):
            for rax in axes:
        
                if isinstance(rax, np.ndarray):
                    fig.colorbar(plot_im, ax=rax, shrink=0.80, location='right');
                else:
                    fig.colorbar(plot_im, ax=rax, shrink=0.80);
        else:
            fig.colorbar(plot_im, ax=axes, shrink=0.80);

    if layout == 'tight':
        fig.tight_layout()

    return fig, axes

def unit_projection(x, figsize=(18,18), unit_size=28, colorbar=True, axes_visible=True):
    """Project the rows of a weight matrix into a square of squares

    Transform a 2-dimensional matrix where each row can be projected onto a 2-dimensional image space into
    a square of squares. E.g. a 784*784 matrix will be transformed to 28 by 28 squares of 28-by-28 pixels where each square is a row in the original matrix.

    Parameters:
        - x: the weight matrix to transform
        - figsize: matplotlib figsize. Default: (18,18)
        - unit_size: size of each square. Default: 28
        - colorbar: display colorbar. Default: True
        - axes_visible: show/hide axes. Default: True
    """

    half = int(0.5*unit_size)
    # reshape
    x = x.flatten().view(unit_size, unit_size, unit_size, unit_size).transpose(1,2).reshape(unit_size*unit_size,unit_size*unit_size).flatten().cpu().detach()

    fig, axes = display(x, size=unit_size*unit_size, figsize=figsize, lims=None, colorbar=colorbar, axes_visible=axes_visible);

    axes.set_xticks(np.arange(0, unit_size*unit_size, unit_size), minor=True)
    axes.set_yticks(np.arange(0, unit_size*unit_size, unit_size), minor=True)
    axes.set_xticks(np.arange(half, unit_size*unit_size+half, unit_size))
    axes.set_yticks(np.arange(half, unit_size*unit_size+half, unit_size))
    axes.set_xticklabels(np.arange(unit_size))
    axes.set_yticklabels(np.arange(unit_size))
    axes.grid(linestyle='--', which='minor')

    return fig, axes

def scatter(x, y, discrete=False, figsize=(8,6), xlabel="", ylabel="", legend=None, figax=None):
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
        axes.scatter(x, y)

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
# Figure 2 and Figure 5B
#
def bootstrap_model_activity(net_list:[ModelState],
                   training_set:Dataset,
                   test_set:Dataset,
                   seq_length=10,
                   seed=2732,
                   lesioned=False,
                   save=True):
    """
    
    Calculates preactivation of models and 
    theoretical bounds (input, data median, category median)
    all CI 95% bootstrapped with replacement

    """
    notn_samples, meds_samples = np.zeros((len(net_list), seq_length)), np.zeros((len(net_list), seq_length))
    gmed_samples, net_samples = np.zeros((len(net_list), seq_length)), np.zeros((len(net_list), seq_length))
    
 
    net_les_samples = np.zeros((len(net_list), seq_length))
    for i, net in enumerate(net_list):
        data, mu_notn, mu_meds, mu_gmed, mu_net, mu_netles =\
        model_activity_lesioned(net, training_set, test_set, seq_length, seed, save)
        # fill sample arrays (model_instances x sequence_length)
        notn_samples[i,:] = mu_notn; meds_samples[i,:]= mu_meds
        gmed_samples[i,:] = mu_gmed; net_samples[i,:] = mu_net
        net_les_samples[i,:] = mu_netles
        
                  
 
    
    # compute bootstrap bounds for each time point 
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles = compute_bootstrap(notn_samples, \
                                                                         meds_samples, \
                                                                             gmed_samples, \
                                                                                 net_samples, net_les_samples)
    # non lesioned netults
    display_model_activity(data, [bs_notn, bs_meds, bs_gmed, bs_net, bs_netles]\
                           ,notn_samples,meds_samples, gmed_samples,net_samples, net_les_samples, save=save)
    # display lesioned results 
    display_model_activity(data, [bs_notn, bs_meds, bs_gmed, bs_net, bs_netles]\
                           ,notn_samples,meds_samples, gmed_samples,net_samples, net_les_samples, lesioned=True, save=save)
        
def compute_bootstrap(notn, meds, gmed, net, net_les=None ,seq_length=10):
    """ compute bootstrap bounds for each time point"""  
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles = [],[],[],[], []
    
    for t in range(seq_length):
        bs_notn.append(bs.bootstrap(notn[:,t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_meds.append(bs.bootstrap(meds[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_gmed.append(bs.bootstrap(gmed[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        bs_net.append(bs.bootstrap(net[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
        
        if net_les is not None:
            bs_netles.append(bs.bootstrap(net_les[:, t], stat_func=bs_stats.mean, iteration_batch_size=None))
            
    return bs_notn, bs_meds, bs_gmed, bs_net, bs_netles

def extract_lower_upper(bs_list):
    """
    wrapper function that extracts upper and lower bounds of the confidence
    interval 
    """
    lower, upper  = [bs.lower_bound for bs in bs_list], [bs.upper_bound for bs in bs_list]
    return lower,upper
    

        
   
def display_model_activity(data,
                           bootstraps,
                           notn_samples,
                           meds_samples,
                           gmed_samples,
                           net_samples, 
                           net_les_samples,
                           lesioned = False,
                           save=True): 
    
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles = bootstraps   
    # create figure plot mean values and 95% CI
    fig, axes = plt.subplots(1, figsize=(14,10))
   
    x = np.arange(1,data.shape[0]+1)
    mu_gmed = np.mean(gmed_samples, axis=0) # empirical mean of global median
    axes.plot(x, mu_gmed, label="dataset median inhibition", color= '0.2')
    lower_gmed, upper_gmed = extract_lower_upper(bs_gmed)
    axes.fill_between(x, lower_gmed, upper_gmed, color='0.2', alpha=0.3) 
    mu_meds = np.mean(meds_samples, axis=0) # empirical mean of category median
    axes.plot(x, mu_meds, label="category median inhibition", color='0.7')
    lower_med, upper_med = extract_lower_upper(bs_meds)
    axes.fill_between(x, lower_med, upper_med, color='0.7', alpha=0.3) 
    
    # add a space between bounds and data that is input-dependent
    axes.plot([],[], linestyle='', label=' ')
    
    if not lesioned: # add input drive to the figure
        mu_notn = np.mean(notn_samples, axis=0) # empirical mean of input drive sequences
        axes.plot(x, mu_notn, label="input", color= '#3388BB')
        lower_notn, upper_notn = extract_lower_upper(bs_notn)
        axes.fill_between(x, lower_notn, upper_notn, color='#3388BB', alpha=0.3) 
    mu_net = np.mean(net_samples, axis=0) # empirical mean of reservoir activity   
    axes.plot(x, mu_net, label="RNN", color= '#EE6666')
    lower_net, upper_net = extract_lower_upper(bs_net)
    axes.fill_between(x, lower_net, upper_net, color='#EE6666', alpha=0.3) 
   
    if lesioned: # add lesioned reservoir to the figure 
        mu_netles = np.mean(net_les_samples, axis=0) # empirical mean of sample set
        axes.plot(x, mu_netles, label="prediction units lesioned", color= '#EECC55')
        lower_netles, upper_netles = extract_lower_upper(bs_netles)
        axes.fill_between(x, lower_netles, upper_netles, color='#EECC55', alpha=0.3) 

    axes.xaxis.set_major_locator(MaxNLocator(integer=True));
  
    axes.legend(fontsize=18,labelspacing=0.1, facecolor='0.95')
    axes.grid(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)
    axes.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)

    if save is True:
        if lesioned:
            save_fig(fig, "lesioned-model-activity", bbox_inches='tight')
        else:
            save_fig(fig, "model-activity", bbox_inches='tight')
            
     
def model_activity(net:ModelState,
                   training_set:Dataset,
                   test_set:Dataset,
                   seq_length=10,
                   seed=2732,
                   save=True):
    """
    calculates model preactivation  and preactivation bounds 
    for unlesioned models 
    """
  
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # category medians and median for all images
    meds = mnist.medians(training_set)
    global_median = training_set.x.median(dim=0).values
    N = 784
    
    with torch.no_grad():
        data, labels = test_set.create_batches(-1, seq_length, shuffle=True)
        data = data.squeeze(0)
        labels = labels.squeeze(0)
        batch_size = data.shape[1]

        # result lists
        mu_notn = [] 
        mu_meds = []
        mu_gmed = []
        mu_net = []
       
        hidden_size = net.model.hidden_size
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
            for i in range(10):
                median[y==i,:] = meds[i]

            # calculate hidden state
            h_meds = (x - median)
            h_gmed = (x - gmedian)
           

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/hidden_size
            m_meds = h_meds.abs().sum(dim=1)/hidden_size
            m_gmed = h_gmed.abs().sum(dim=1)/hidden_size
            
                

            h_net, l_net = net.model(x, state=h_net) 
            m_net = torch.cat([a for a in l_net], dim=1).abs().mean(dim=1).mean()
        
                
            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_net.append(m_net.mean().cpu().item())
            


        return data, np.array(mu_notn), np.array(mu_meds), np.array(mu_gmed), np.array(mu_net)
      



#
# Figure 3 
#
def example_sequence_state(net:ModelState, dataset:Dataset, save=True):
    batches, _ = dataset.create_batches(batch_size=1, sequence_length=10, shuffle=False)

    seq = batches[0,:,:,:]

    X = []; P = []; H = []

    h = net.model.init_state(seq.shape[1])
 
    for x in seq:

        p = net.predict(h)
        h, l_a = net.model(x, state=h) # EDITED: old: h, [a,b] new: [a,b,c]

        X.append(x.mean(dim=0).detach().cpu())
        P.append(p.mean(dim=0).detach().cpu())
        H.append(h.mean(dim=0).detach().cpu())


    fig = plt.figure(figsize=(3,3))
   
    fig, axes = display(X+P,  shape=(10,2), figsize=(3,3), axes_visible=False, layout='tight')
  
    
    if save is True:
        save_fig(fig, "example_sequence_state", bbox_inches='tight')
        
def plot_colorbars(im1, im2, save=True):
    """
    Workaround for color bars figures since current display function
    does not properly display multiple color bars

    Parameters
    ----------
    im1 : Input drive image.
    im2 : Internal drive image .

    Returns
    -------
    None.

    """
    fig, (ax1, ax2) = plt.subplots(2,1)
    cmap1 = truncate_colormap('seismic',0.5, 1)
    cmap2 = truncate_colormap('seismic',-1, 0.8705)
    mi,ma = im2.min(), im2.max()

    im1 = ax1.imshow(im1,cmap=cmap1)
    im2 =  ax2.imshow(im2,cmap=cmap2, vmin=mi, vmax=ma)
    bbox_ax_top = ax1.get_position()
    bbox_ax_bottom = ax2.get_position()
    cbar_ax1, cbar_ax2 = fig.add_axes([bbox_ax_top.x1 + 0.1 , bbox_ax_top.y1, 0.02, bbox_ax_top.y1 - bbox_ax_top.y0]),\
    fig.add_axes([bbox_ax_bottom.x1 + 0.1, bbox_ax_bottom.y1, 0.02,  bbox_ax_bottom.y1 - bbox_ax_bottom.y0])
    fig.colorbar(im2,cax=cbar_ax2, cmap=cmap2, ticks=[-1, -0.75, -0.5, -0.25, 0, 0.25])
    fig.colorbar(im1, cax=cbar_ax1, cmap=cmap1)
    
    if save is True:
        save_fig(fig, "drive", bbox_inches='tight')

    


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



def _calc_xdrive_pdrive(net:ModelState, dataset:Dataset):
    """Calculates the excitatory presynaptic activity from input (xdrive) and recurrent connections (pdrive)
    """
    steps = 10
    b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=False)
    batch = b.squeeze(0)

    h = net.model.init_state(1)
    for i in range(steps):
        x = batch[i]
        p = net.predict(h)
        h, l_a = net.model(x, state=h)

    pdrive = F.relu(p).mean(dim=0).detach()
    xdrive = F.relu(x).mean(dim=0).detach()
  
    return xdrive, pdrive

def _pred_mask(net:ModelState, dataset:Dataset):
    """Creates a mask that can be used to turn off the prediction units in network,
    based on whether there is higher excitatory presynaptic activity from recurrent units as opposed to input units
    """
    xdrive, pdrive = _calc_xdrive_pdrive(net, dataset)
    xpdrive = (xdrive-pdrive)
    pred_mask = torch.ones(28*28)
    pred_mask[xpdrive<0] = 0

    return pred_mask

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def xdrive_pdrive(net:ModelState, dataset:Dataset, save=True):
    """
    determines whether a unit is driven by the input or reciprocal activity
    """
    xdrive, pdrive = _calc_xdrive_pdrive(net, dataset)

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2, 3)
    cmap_base = 'seismic'
    vmin, vmax = 0.5, 1
    cmap = truncate_colormap(cmap_base, vmin, vmax)
    
    ax = fig.add_subplot(gs[:,0:-1])
    scatter(xdrive,pdrive,figax=(fig,ax))
   
    
    ax.grid(True)
 
    ax = fig.add_subplot(gs[0,2])
    
   
    display(pdrive,lims=(0, pdrive.max()),cmap = cmap,colorbar=True,figax=(fig,ax));
  
    ax.set_xticks([])
    ax.set_yticks([])
  
    ax = fig.add_subplot(gs[1,2])
   

    display(xdrive,lims=(0, xdrive.max()),cmap=  cmap,colorbar=True,figax=(fig,ax));

    ax.set_xticks([])
    ax.set_yticks([])
   

    if save is True:
        save_fig(fig, net.title+"/xdrive_pdrive", bbox_inches='tight')
        


def prediction_units(net:ModelState, dataset:Dataset, save=True):
    xdrive, pdrive = _calc_xdrive_pdrive(net, dataset)
    xpdrive = (xdrive-pdrive)

    predunits = torch.ones(28*28)*-1
    predunits[xpdrive<0] = 1
    fig, axes = display(predunits,axes_visible=False,cmap='binary');

    if save is True:
        save_fig(fig, net.title+"/prediction-units")

def model_activity_lesioned(net:ModelState, training_set:Dataset, test_set:Dataset, seq_length=10, seed=2553, save=True):
    """
    calculates model preactivation  and preactivation bounds 
    for lesioned models 
    """
    # create mask that knocks out predictive units
    mask = _pred_mask(net, test_set)
   
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # category medians and median for all images
    meds = mnist.medians(training_set)
    global_median = training_set.x.median(dim=0).values

    with torch.no_grad():
        data, labels = test_set.create_batches(-1, seq_length, shuffle=True)
        data = data.squeeze(0)
        labels = labels.squeeze(0)
        batch_size = data.shape[1]

        # result lists
        mu_notn = []; 
        mu_meds = [];
        mu_gmed = []; 
        mu_net = []; 
        mu_netles = []; 


        h_net = net.model.init_state(batch_size)
        h_netles = net.model.init_state(batch_size)
        
       
        for i in range(data.shape[0]):
            x = data[i]
            y = labels[i]

            # repeat global median for each input image
            gmedian = torch.zeros_like(x)
            gmedian[:,:] = global_median
            # find the corresponding median for each input image
            median = torch.zeros_like(x)
            for i in range(10):
                median[y==i,:] = meds[i]

            # calculate hidden state
            h_meds = (x - median)
            h_gmed = (x - gmedian)
            h_net, l_net = net.model(x, state=h_net)
            h_netles = h_netles * mask # perform lesion
            h_netles, l_netles = net.model(x, state=h_netles)

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/net.model.hidden_size
            m_meds = h_meds.abs().sum(dim=1)/net.model.hidden_size
            m_gmed = h_gmed.abs().sum(dim=1)/net.model.hidden_size
            m_net = torch.cat([a for a in l_net], dim=1).abs().mean(dim=1) # EDITED: old expr: 'torch.cat([a for a in l_tub], dim=1).abs().mean(dim=1)
            m_netles = torch.cat([a for a in l_netles], dim=1).abs().mean(dim=1)
      

            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_net.append(m_net.mean())
            mu_netles.append(m_netles.mean())

         
           
    return data, np.array(mu_notn), np.array(mu_meds), np.array(mu_gmed), np.array(mu_net), np.array(mu_netles)

#
# Appendix
#

def weights_mean_activity(net:ModelState, dataset:Dataset, save=True):
    def Whmean(steps):
        # calculates mean network activity after certain amount of timesteps
        # and times this by the weight matrix to get the mean presynaptic activity of the recurrent connections
        b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=True)
        batch = b.squeeze(0)

        h = net.model.init_state(1)
        for i in range(steps):
            h, l_a = net.model(batch[i], state=h)

        return (h.mean(dim=0) * net.model.W.t()).t().detach().cpu()

    # Weights times mean activity at timestep 2
    fig, axes = unit_projection(Whmean(2), colorbar=False, axes_visible=False);
    if save is True:
        save_fig(fig, net.title+"/weights_mean_activity_t2", bbox_inches='tight')

    # Weights times mean activity at timestep 9
    fig,axes = unit_projection(Whmean(9), colorbar=False, axes_visible=False);
    if save is True:
        save_fig(fig, net.title+"/weights_mean_activity_t9", bbox_inches='tight')


def pred_after_timestep(net:ModelState, dataset:Dataset, mask=None, save=True):
    imgs= []

    for d in range(10):
        imgs = imgs + [net.predict(torch.zeros(1,784))] + [net.predict(_run_seq_from_digit(d, i, net, dataset, mask=mask)).mean(dim=0) for i in range(1,10)]

    fig, axes = display(imgs, shape=(10,10), axes_visible=False)

    for i in range(10):
        axes[i][0].set_ylabel(str(i)+":", ha='center', fontsize=60, rotation=0, labelpad=30)
   

    fig.tight_layout()

    if save:
        save_fig(fig,
                      net.title+"/pred-after-timestep" + ("-lesioned" if mask is not None else ""),
                      bbox_inches='tight')
        

    
    