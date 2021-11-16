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

def save_fig(fig, name, bbox_inches=None):
    """Convenience wrapper for saving figures in a default "../figures/" directory and auto appends file extensions ".svg"
    and ".png"
    """
    filepath = "Figures/" + name
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
def bootstrap_model_activity(net_list:[ModelState],
                   training_set:Dataset,
                   test_set:Dataset,
                   seq_length=10,
                   lesioned=True,
                   latent=False,
                   seed=None,
                   save=True,
                   data_type='mnist'):
    """
    
    Calculates preactivation of models and 
    theoretical bounds (input, data median, category median)
    all CI 95% bootstrapped with replacement

    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # fill samples for input (notn) and category median (meds)
    notn_samples, meds_samples = np.zeros((len(net_list), seq_length)), np.zeros((len(net_list), seq_length))
    # fill samples for global median (gmed) and network preactivation (net)
    gmed_samples, net_samples = np.zeros((len(net_list), seq_length)), np.zeros((len(net_list), seq_length))
    
    # fill samples for lesioned prediction units network (net_les)
    net_les_samples = np.zeros((len(net_list), seq_length))
    # fill samples for lesioned error units network (net_les_rev)
    net_les_samples_rev = np.zeros((len(net_list), seq_length))
    for i, net in enumerate(net_list):
        # calculate preactivation curves when prediction units are lesioned
        data, mu_notn, mu_meds, mu_gmed, mu_net, mu_netles, _, _ =\
        helper.model_activity_lesioned(net, training_set, test_set, seq_length, save,\
                                latent=latent, data_type=data_type)
        
        # calculate preactivation curves when error units are lesioned (control)
        data_rev, mu_notn_rev, mu_meds_rev, mu_gmed_rev, mu_net_rev, mu_netles_rev, _, _ =\
        helper.model_activity_lesioned(net, training_set, test_set, seq_length, save,\
                                latent=latent, data_type=data_type, reverse=True)
            
        # fill sample arrays (model_instances x sequence_length)
        notn_samples[i,:] = mu_notn; meds_samples[i,:]= mu_meds
        gmed_samples[i,:] = mu_gmed; net_samples[i,:] = mu_net
        net_les_samples[i,:] = mu_netles
        net_les_samples_rev[i,:] = mu_netles_rev
                  
 
    
    # compute bootstrap bounds for each time point 
    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev = helper.compute_bootstrap(notn_samples, \
                                                                         meds_samples, \
                                                                             gmed_samples, \
                                                                                 net_samples, net_les_samples, net_les_samples_rev)
    # non lesioned netults
    if not lesioned:
        display_model_activity(data, [bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev]\
                              ,notn_samples,meds_samples, gmed_samples,net_samples, net_les_samples, net_les_samples_rev, save=save, data_type=data_type)
            
    # display lesioned results (includes control)
    else:
        display_model_activity(data, [bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev]\
                                ,notn_samples,meds_samples, gmed_samples,net_samples, net_les_samples, net_les_samples_rev, lesioned=True, save=save, reverse=True, data_type=data_type)




def display_model_activity(data,
                           bootstraps,
                           notn_samples,
                           meds_samples,
                           gmed_samples,
                           net_samples, 
                           net_les_samples,
                           net_les_samples_rev,
                           lesioned = False,
                           save=True,
                           reverse=False,
                           data_type='mnist'): 
    """
    visualises preactivation of models and preactivations derived from
    theoretical bounds including confidence intervals
    """

    bs_notn, bs_meds, bs_gmed, bs_net, bs_netles, bs_netles_rev = bootstraps   
    # create figure plot mean values and 95% CI
    fig, axes = plt.subplots(1, figsize=(14,10))
   
    x = np.arange(1,data.shape[0]+1)
    mu_gmed = np.mean(gmed_samples, axis=0) # empirical mean of global median
    axes.plot(x, mu_gmed, label="dataset median inhibition", color= '0.2')
    lower_gmed, upper_gmed = helper.extract_lower_upper(bs_gmed)
    axes.fill_between(x, lower_gmed, upper_gmed, color='0.2', alpha=0.3) 
    mu_meds = np.mean(meds_samples, axis=0) # empirical mean of category median
    axes.plot(x, mu_meds, label="category median inhibition", color='0.7')
    lower_med, upper_med = helper.extract_lower_upper(bs_meds)
    axes.fill_between(x, lower_med, upper_med, color='0.7', alpha=0.3) 
    
    # add a space between bounds and data that is input-dependent
    axes.plot([],[], linestyle='', label=' ')
    
    if not lesioned: # add input drive to the figure
        mu_notn = np.mean(notn_samples, axis=0) # empirical mean of input drive sequences
        axes.plot(x, mu_notn, label="input", color= '#3388BB')
        lower_notn, upper_notn = helper.extract_lower_upper(bs_notn)
        axes.fill_between(x, lower_notn, upper_notn, color='#3388BB', alpha=0.3) 
    mu_net = np.mean(net_samples, axis=0) # empirical mean of reservoir activity   
    axes.plot(x, mu_net, label="RNN", color= '#EE6666')
    lower_net, upper_net = helper.extract_lower_upper(bs_net)

    axes.fill_between(x, lower_net, upper_net, color='#EE6666', alpha=0.3) 
   
    if lesioned: # add lesioned reservoir to the figure 
        mu_netles = np.mean(net_les_samples, axis=0) # empirical mean of sample set
        axes.plot(x, mu_netles, label="prediction units lesioned", color= '#EECC55')
        lower_netles, upper_netles = helper.extract_lower_upper(bs_netles)
        axes.fill_between(x, lower_netles, upper_netles, color='#EECC55', alpha=0.3) 
        if reverse:
            mu_netles_rev = np.mean(net_les_samples_rev, axis=0) # empirical mean of sample set
            axes.plot(x, mu_netles_rev, label="error units lesioned", color= '#5efc03')
            lower_netles_rev, upper_netles_rev = helper.extract_lower_upper(bs_netles_rev)
        
            axes.fill_between(x, lower_netles_rev, upper_netles_rev, color='#5efc03', alpha=0.3) 

    axes.xaxis.set_major_locator(MaxNLocator(integer=True));
  
    axes.legend(fontsize=18,labelspacing=0.1, facecolor='0.95')
    axes.grid(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.xaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)
    axes.yaxis.set_tick_params(which='major', size=10, width=2, labelsize=16)

    if save is True:
        if lesioned:
            save_fig(fig, "preactivation_curves" + data_type+"/lesioned-model-activity", bbox_inches='tight')
        else:
            save_fig(fig,  "preactivation_curves" + data_type+"/model-activity", bbox_inches='tight')
    return fig  

#
# Figure 2B 
#
def example_sequence_state(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=True):
    """
    visualises input and internal drive for a sample sequence
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    batches, _ = dataset.create_batches(batch_size=1, sequence_length=10, shuffle=False)

    seq = batches[0,:,:,:]
    input_size = seq.shape[-1] # make sure we only visualize input units and no latent resources
    X = []; P = []; L=[]
    
    h = net.model.init_state(seq.shape[1])
 
    for x in seq:

        p = net.predict(h, latent)
        h, l_a = net.model(x, state=h)
      
   
        X.append(x[:input_size].mean(dim=0).detach().cpu())
        P.append(p[:,:input_size].mean(dim=0).detach().cpu())
        if latent: # look at latent unit drive 
            L.append(p[:,input_size:].mean(dim=0).detach().cpu())

    fig = plt.figure(figsize=(3,3))
    if latent:
        fig, axes = display(X+P+L,  shape=(10,3), figsize=(3,3), axes_visible=False, layout='tight')
    else:
         fig, axes = display(X+P,  shape=(10,2), figsize=(3,3), axes_visible=False, layout='tight')

    
    if save is True:
        save_fig(fig, "example_sequence_state", bbox_inches='tight')

#
# Figure 3B
#
def topographic_distribution(net:ModelState, dataset:Dataset, training_set:Dataset=None, seed=2553, save=False):
    """
    
    plots topographic distribution of prediction and error units in
    data space.
    
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    preact_stats = helper.compute_preact_stats(net, dataset)
    nunits = net.model.hidden_size
    pred_rule = torch.zeros(nunits)
    count = 0
    seen = []
    for cls_plt in range(10):
        med, mad = preact_stats[:, cls_plt, 0], preact_stats[:, cls_plt, 1]
        
        # scale MAD to obtain a pseudo standard deviation
        # https://stats.stackexchange.com/questions/355943/scale-factor-for-mad-for-non-normal-distribution)
        
        for i in range(nunits): # 99% CI
            if (torch.abs(med[i]) - torch.abs(2.576*mad[i])) > 0:
                pred_rule[i] = 0.5
                if i not in seen:
                    count += 1
                    seen.append(i)

    for i in range(nunits): # prediction unit
        if pred_rule[i] < 0.5:
            pred_rule[i] = -0.5

                
    fig,ax  = plt.subplots(1, 1)

    cmap_base = 'seismic'
    vmin, vmax = 0, 1
    cmap = helper.truncate_colormap(cmap_base, vmin, vmax)

    display(pred_rule, axes_visible=False, cmap=cmap, figax=(fig,ax))
    if save is True:
        save_fig(fig, net.title+"/topo_distrib_", bbox_inches='tight')
#
# Figure 3A, 5B & Appendix A Figure A3 & A4
#     
def scatter_units(net:ModelState, dataset:Dataset, training_set:Dataset=None, cls_plt=9, seed=2553, save=False):
    """
    Create scatter plots of the units
    Units will be assigned to two colors, red for prediction units
    & blue for error units
    Decision boundary is plotted that demarcates the border between
    prediction & error units.

    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    preact_stats = helper.compute_preact_stats(net, dataset)
  
    med, mad = preact_stats[:, cls_plt, 0], preact_stats[:, cls_plt, 1]
      
 
    fig,ax  = plt.subplots(1, 1)
    nunits = net.model.hidden_size
    pred_rule = torch.zeros(nunits)
    # scale MAD to obtain a pseudo standard deviation
    # https://stats.stackexchange.com/questions/355943/scale-factor-for-mad-for-non-normal-distribution)
    for i in range(nunits):
       
        if (torch.abs(med[i]) - torch.abs(2.576*mad[i])) > 0:
            pred_rule[i] = 1 # prediction unit is predictive for class 'cls_plt'

    
    scatter(mad[pred_rule>0] ,med[pred_rule>0],color='r',figax=(fig,ax))
    
    scatter(mad[pred_rule==0], med[pred_rule==0],color='b',figax=(fig,ax))

    # construct decision boundary for 
    d_bound = 2.576*np.linspace(mad.min(), mad.max(), len(mad))
    ax.plot(np.linspace(mad.min(), mad.max(), len(mad)), d_bound, '--', color='black')

    ax.set_xlabel("MAD of preactivation at final time step",fontsize=13)
    ax.set_ylabel("Median preactivation at final time step",fontsize=13);
    ax.grid(True)

    if save is True:
        save_fig(fig, net.title+"/decision_boundary"+str(cls_plt), bbox_inches='tight')
        

#
# Appendix A Figures A1 & A2
#

def pred_after_timestep(net:ModelState, dataset:Dataset, mask=None, seed=2553, save=True):
    """
    visualises internal drive after 0-9 preceding frames.
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    imgs= []

    for d in range(10):
        imgs = imgs + [net.predict(torch.zeros(1,784))] + [net.predict(helper._run_seq_from_digit(d, i, net, dataset, mask=mask)).mean(dim=0) for i in range(1,10)]

    fig, axes = display(imgs, shape=(10,10), axes_visible=False)

    for i in range(10):
        axes[i][0].set_ylabel(str(i)+":", ha='center', fontsize=60, rotation=0, labelpad=30)
   

    fig.tight_layout()

    if save:
        save_fig(fig,
                      net.title+"/pred-after-timestep" + ("-lesioned" if mask is not None else ""),
                      bbox_inches='tight')
        

    
    