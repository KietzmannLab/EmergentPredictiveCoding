import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cycler
from sklearn.manifold import MDS, TSNE
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

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
plt.rc('errorbar', capsize=4)
# for a bit nicer font in plots
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.size'] = 12



#
# ---------------     Convenience functions     ---------------
#

def save_fig(fig, name, bbox_inches=None):
    """Convienience wrapper for saving figures in a default "../figures/" directory and auto appends file extension ".png"
    """
    filepath = "../figures/" + name
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

        if size is None:
            _size = int(np.sqrt(img.size))
            img = img[:_size*_size].reshape(_size,_size)
        else:
            img = img[:size[0]*size[1]].reshape(size[0],size[1])

        plot_im = ax.imshow(img, cmap=cmap)

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


#
# ---------------     Plots from the thesis     ---------------
#

#
# Section 3.4
#

def example_mnist_sequences(test_set:Dataset, save=True):
    data, labels = test_set.create_batches(-1, 5, shuffle=False)
    data = data.squeeze(0)
    fig, axes = display(
        [data[i,0,:] for i in range(5)] +
        [data[i,7,:] for i in range(5)]
        ,layout='tight'
        ,axes_visible=False);
    fig.text(0.5, -0.04, r"time $\longrightarrow$", ha='center', fontsize=30);
    axes[0][0].set_ylabel(r"1:", ha='center', fontsize=30, rotation=0, labelpad=30)
    axes[1][0].set_ylabel(r"2:", ha='center', fontsize=30, rotation=0, labelpad=30)
    if save is True:
        save_fig(fig, "/mnist-misc/example-sequences", bbox_inches='tight')

def mnist_medians(training_set:Dataset, save=True):
    meds = mnist.medians(training_set)

    fig, axes = display([meds[i] for i in range(10)], shape=(5,2), axes_visible=False)
    for i,ax in enumerate(axes.flat):
        ax.set_xlabel(str(i), fontsize=30, labelpad=5)
    fig.tight_layout()
    if save is True:
        save_fig(fig, "/mnist-misc/mnist-medians", bbox_inches='tight')

def mnist_median(training_set:Dataset, save=True):
    fig, axes = display(training_set.x.median(dim=0).values, axes_visible=False);
    if save is True:
        save_fig(fig, "../figures/mnist-misc/mnist-median")

def mnist_sum(training_set:Dataset, save=True):
    s = training_set.x.sum(dim=0)
    s[training_set.x.sum(dim=0) > 1] = 1
    fig, axes = display(s, axes_visible=False);
    if save is True:
        save_fig(fig, "../figures/mnist-misc/mnist-sum")

#
# Section 4.1
#

def training_progress(lstm:ModelState, rnn:ModelState, tub:ModelState, save=True):
    fig, axes = init_axes(4, figsize=(6,8), shape=(2,2))

    x = np.arange(1, 201)

    axes[0][0].plot(x, lstm.results["train loss"], label="Training set")
    axes[0][0].plot(x, lstm.results["test loss"], label="Test set")

    axes[0][1].plot(x, rnn.results["train loss"], label="Training set")
    axes[0][1].plot(x, rnn.results["test loss"], label="Test set")

    axes[1][0].plot(x, tub.results["train loss"], label="Training set")
    axes[1][0].plot(x, tub.results["test loss"], label="Test set")

    axes[1][1].plot(x, lstm.results["train loss"], label="LSTM")
    axes[1][1].plot(x, rnn.results["train loss"], label="SRN")
    axes[1][1].plot(x, tub.results["train loss"], label="Bathtub")

    for ax, label in zip(axes.flat, [('a','LSTM'),('b','SRN'),('c','Bathtub'),('d','All')]):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));
        ax.set_xlabel(r"Training time $\longrightarrow$",fontsize=16)
        ax.set_ylabel("Loss",fontsize=16)
        ax.legend()
        ax.set_title("("+label[0]+") "+label[1], fontsize=18)
        ax.grid()

    fig.tight_layout()

    if save is True:
        save_fig(fig, "training-progress", bbox_inches='tight')

def _t_test(m1, m2):
    """Convenience wrapper around scipy.stat.ttest_ind"""
    t, p = ttest_ind(m1, m2, equal_var = False)
    return p

def model_activity(lstm:ModelState,
                   rnn:ModelState,
                   tub:ModelState,
                   training_set:Dataset,
                   test_set:Dataset,
                   seq_length=9,
                   seed=2732,
                   save=True):
    # Use a seed for reproducability
    if seed is None:
        seed = np.random.randint(1e4)
        print(seed)
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
        mu_notn = []; sig_notn = []
        mu_meds = []; sig_meds = []
        mu_gmed = []; sig_gmed = []
        mu_lstm = []; sig_lstm = []
        mu_rnn = []; sig_rnn = []
        mu_tub = []; sig_tub = []

        p_values = []

        h_lstm = lstm.model.init_state(batch_size)
        h_rnn = rnn.model.init_state(batch_size)
        h_tub = tub.model.init_state(batch_size)

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
            h_lstm, l_lstm = lstm.model(x, state=h_lstm)
            h_rnn, l_rnn = rnn.model(x, state=h_rnn)
            h_tub, l_tub = tub.model(x, state=h_tub)

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/tub.model.hidden_size
            m_meds = h_meds.abs().sum(dim=1)/tub.model.hidden_size
            m_gmed = h_gmed.abs().sum(dim=1)/tub.model.hidden_size
            m_lstm = torch.cat([a for a in l_lstm], dim=1).abs().mean(dim=1)
            m_rnn = torch.cat([a for a in l_rnn], dim=1).abs().mean(dim=1)
            m_tub = torch.cat([a for a in l_tub], dim=1).abs().mean(dim=1)

            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_lstm.append(m_lstm.mean().cpu().item())
            mu_rnn.append(m_rnn.mean().cpu().item())
            mu_tub.append(m_tub.mean().cpu().item())

            # Calculate the standard error of the mean
            sig_notn.append(m_notn.std().cpu().item()/np.sqrt(batch_size))
            sig_meds.append(m_meds.std().cpu().item()/np.sqrt(batch_size))
            sig_gmed.append(m_gmed.std().cpu().item()/np.sqrt(batch_size))
            sig_lstm.append(m_lstm.std().cpu().item()/np.sqrt(batch_size))
            sig_rnn.append(m_rnn.std().cpu().item()/np.sqrt(batch_size))
            sig_tub.append(m_tub.std().cpu().item()/np.sqrt(batch_size))

            # Perform welch's t-test on relevant adjacent pairs
            p_values.append([
                _t_test(m_gmed, m_tub),
                _t_test(m_tub, m_rnn),
                _t_test(m_rnn, m_lstm),
                _t_test(m_lstm, m_meds)
            ])


        fig, axes = plt.subplots(1, figsize=(14,10))

        x = np.arange(1,data.shape[0]+1)

        axes.errorbar(x, mu_notn, yerr=sig_notn, capsize=4, label="input")
        axes.errorbar(x, mu_meds, yerr=sig_meds, capsize=4, label="cat. median")
        axes.errorbar(x, mu_gmed, yerr=sig_gmed, capsize=4, label="median")
        axes.errorbar(x, mu_tub, yerr=sig_tub, capsize=4, label="Bathtub")
        axes.errorbar(x, mu_lstm, yerr=sig_lstm, capsize=4, label="LSTM")
        axes.errorbar(x, mu_rnn, yerr=sig_rnn, capsize=4, label="SRN")

        # Plot asterisks for statistical significance between adjacent pairs of means
        def sig_asterix(m1, m2, p):
            y = (m1 - m2) / 2 + m2
            if p < 0.001:
                axes.text(i+1, y-0.0005, r'***', fontsize=16)
            elif p < 0.01:
                axes.text(i+1, y-0.0005, r'**', fontsize=16)
            elif p < 0.05:
                axes.text(i+1, y-0.0005, r'*', fontsize=16)
            else:
                axes.text(i+1, y-0.0005, r'$-$', fontsize=16)
        for i in range(1,9):
            sig_asterix(mu_gmed[i], mu_tub[i], p_values[i][0])
            sig_asterix(mu_tub[i], mu_rnn[i], p_values[i][1])
            sig_asterix(mu_rnn[i], mu_lstm[i], p_values[i][2])
            sig_asterix(mu_lstm[i], mu_meds[i], p_values[i][3])

        axes.xaxis.set_major_locator(MaxNLocator(integer=True));
        axes.set_xlabel(r"time $\longrightarrow$",fontsize=16)
        axes.set_ylabel("Mean absolute activity",fontsize=16)
        axes.legend(fontsize=16)
        axes.grid()
        fig.tight_layout()

        if save is True:
            save_fig(fig, "model-activity", bbox_inches='tight')

#
# Section 4.2
#

def feedback_hist(lstm:ModelState,
                  lstm_u:ModelState,
                  rnn:ModelState,
                  rnn_u:ModelState,
                  tub:ModelState,
                  tub_u:ModelState,
                  dataset:Dataset,
                  save=True):
    data, labels = dataset.create_batches(-1, 1, shuffle=True)
    x = data.squeeze()

    h_lstm, _ = lstm.model(x)
    h_lstmu, _ = lstm_u.model(x)
    h_rnn, _ = rnn.model(x)
    h_rnnu, _ = rnn_u.model(x)

    p_lstm = lstm.predict(h_lstm).mean(dim=0).detach()
    p_lstmu = lstm_u.predict(h_lstmu).mean(dim=0).detach()
    p_rnn = rnn.predict(h_rnn).mean(dim=0).detach()
    p_rnnu = rnn_u.predict(h_rnnu).mean(dim=0).detach()


    fig, axes = plt.subplots(2,2, figsize=(10,10), sharey='row', sharex='row')

    axes[0][0].hist(p_lstmu, bins=8);
    axes[0][1].hist(p_lstm, bins=30, color='#3388BB');
    axes[0][0].set_ylabel("Frequency")

    axes[1][0].hist(p_rnnu, bins=25);
    axes[1][1].hist(p_rnn, bins=25, color='#3388BB');
    axes[1][0].set_ylabel("Frequency")

    for ax in axes.flat:
        ax.grid()
        ax.set_xlabel(r"p")
    for ax in axes:
        ax[0].set_title(r"before training",fontsize=15)
        ax[1].set_title(r"after training",fontsize=15)

    fig.text(0.5, 1.0, r"(a) LSTM", ha='center', fontsize=15);
    fig.text(0.5, 0.5, r"(b) SRN", ha='center', fontsize=15);

    fig.tight_layout()

    if save is True:
        save_fig(fig, "feedback-hist", bbox_inches='tight')

#
# Section 4.3
#

def example_sequence_state(ms:ModelState, dataset:Dataset, save=True):
    batches, _ = dataset.create_batches(batch_size=1, sequence_length=9, shuffle=False)

    seq = batches[1,:,:,:]

    X = []; P = []; H = []

    h = ms.model.init_state(seq.shape[1])
    for x in seq:

        p = ms.predict(h)
        h, [a] = ms.model(x, state=h)

        X.append(x.mean(dim=0).detach().cpu())
        P.append(p.mean(dim=0).detach().cpu())
        H.append(h.mean(dim=0).detach().cpu())


    fig, axes = display(X+P+H, shape=(9,3), figsize=(3,3), axes_visible=False, layout='tight')

    fig.text(0.5, -0.03, r"time (t) $\longrightarrow$", ha='center', fontsize=30);
    axes[0][0].set_ylabel(r"$\mathbf{x}_t$", ha='center', fontsize=30, rotation=0, labelpad=30)
    axes[1][0].set_ylabel(r"$\mathbf{p}_t$", ha='center', fontsize=30, rotation=0, labelpad=30)
    axes[2][0].set_ylabel(r"$\mathbf{h}_t$", ha='center', fontsize=30, rotation=0, labelpad=30)

    if save is True:
        save_fig(fig, ms.title+"/example_sequence_state", bbox_inches='tight')

def _run_seq_from_digit(digit, steps, ms:ModelState, dataset:Dataset, mask=None):
    """Create sequences with the same starting digit through a model and return the hidden state

    Parameters:
        - digit: the last digit in the sequence
        - steps: sequence length, or steps before the sequence gets to the 'digit'
        - ms: model
        - dataset: dataset to use
        - mask: mask can be used to turn off (i.e. lesion) certain units
    """
    fixed_starting_point = (digit - steps) % 10
    b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=True, fixed_starting_point=fixed_starting_point)
    batch = b.squeeze(0)

    h = ms.model.init_state(1)
    for i in range(steps):
        h, l_a = ms.model(batch[i], state=h)
        if mask is not None:
            h = h * mask

    return h.detach()

def difference_h_t2_vs_t9(ms:ModelState, dataset:Dataset, mask=None, save=True):
    h2s = []; h9s = []; difs = [];
    for i in range(10):
        h2 = _run_seq_from_digit((i+1)%10, 2, ms, dataset, mask=mask).cpu().median(dim=0).values
        h9 = _run_seq_from_digit((i+1)%10, 9, ms, dataset, mask=mask).cpu().median(dim=0).values
        h2s.append(h2)
        h9s.append(h9)
        difs.append(h9-h2)
    fig, axes = display(h2s + h9s + difs, shape=(10,3), colorbar=False, axes_visible=False);

    axes[0][0].set_ylabel(r"$\mathbf{h}_{t=2}$", ha='center', fontsize=40, rotation=0, labelpad=45)
    axes[1][0].set_ylabel(r"$\mathbf{h}_{t=9}$", ha='center', fontsize=40, rotation=0, labelpad=45)
    axes[2][0].set_ylabel("diff.", ha='center', fontsize=35, rotation=0, labelpad=45)

    for i in range(10):
        axes[2][i].set_xlabel(str(i), ha='center', fontsize=40, rotation=0, labelpad=5)

    fig.tight_layout()

    if save:
        save_fig(fig,
                      ms.title+"/difference-h-t2-vs-t9" + ("-lesioned" if mask is not None else ""),
                      bbox_inches='tight')

def _calc_xdrive_pdrive(ms:ModelState, dataset:Dataset):
    """Calculates the excitatory presynaptic activity from input (xdrive) and recurrent connections (pdrive)
    """
    steps = 9
    b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=False)
    batch = b.squeeze(0)

    h = ms.model.init_state(1)
    for i in range(steps):
        x = batch[i]
        p = ms.predict(h)
        h, l_a = ms.model(x, state=h)

    pdrive = F.relu(p).mean(dim=0).detach()
    xdrive = F.relu(x).mean(dim=0).detach()

    return xdrive, pdrive

def _pred_mask(ms:ModelState, dataset:Dataset):
    """Creates a mask that can be used to turn off the prediction units in a bathtub model,
    based on whether there is higher excitatory presynaptic activity from recurrent units as opposed to input units
    """
    xdrive, pdrive = _calc_xdrive_pdrive(ms, dataset)
    xpdrive = (xdrive-pdrive)
    pred_mask = torch.ones(28*28)
    pred_mask[xpdrive<0] = 0

    return pred_mask

def xdrive_pdrive(ms:ModelState, dataset:Dataset, save=True):
    xdrive, pdrive = _calc_xdrive_pdrive(ms, dataset)

    fig = plt.figure(figsize=(12,6))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0,0])
    display(pdrive,lims=None,colorbar=True,figax=(fig,ax));
    ax.set_xlabel(r"$\langle \mathbf{p}^+ \rangle$",fontsize=15)
    ax.set_title("(a)", fontsize=18)
    ax = fig.add_subplot(gs[1,0])
    display(xdrive,lims=None,colorbar=True,figax=(fig,ax));
    ax.set_xlabel(r"$\langle \mathbf{x}^+ \rangle$",fontsize=15)
    ax.set_title("(b)", fontsize=18)

    ax = fig.add_subplot(gs[:,1:])
    scatter(xdrive,pdrive,figax=(fig,ax))
    ax.set_xlabel(r"$\langle \mathbf{x}^+ \rangle$",fontsize=15)
    ax.set_ylabel(r"$\langle \mathbf{p}^+ \rangle$",fontsize=15);
    ax.set_title("(c)", fontsize=18)

    fig.tight_layout()

    if save is True:
        save_fig(fig, ms.title+"/xdrive_pdrive", bbox_inches='tight')

def prediction_units(ms:ModelState, dataset:Dataset, save=True):
    xdrive, pdrive = _calc_xdrive_pdrive(ms, dataset)
    xpdrive = (xdrive-pdrive)

    predunits = torch.ones(28*28)*-1
    predunits[xpdrive<0] = 1
    fig, axes = display(predunits,axes_visible=False,cmap='binary');

    if save is True:
        save_fig(fig, ms.title+"/prediction-units")

def model_activity_lesioned(ms:ModelState, training_set:Dataset, test_set:Dataset, seq_length=9, seed=2553, save=True):
    mask = _pred_mask(ms, test_set)

    # Use a seed for reproducability
    if seed is None:
        seed = np.random.randint(1e4)
        print(seed)
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
        mu_notn = []; sig_notn = []
        mu_meds = []; sig_meds = []
        mu_gmed = []; sig_gmed = []
        mu_tub = []; sig_tub = []
        mu_tubles = []; sig_tubles = []

        p_values = []

        h_tub = ms.model.init_state(batch_size)
        h_tubles = ms.model.init_state(batch_size)

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
            h_tub, l_tub = ms.model(x, state=h_tub)
            h_tubles = h_tubles * mask # perform lesion
            h_tubles, l_tubles = ms.model(x, state=h_tubles)

            # calculate L1 loss for each unit, assuming equal amounts of units in each model
            m_notn = x.abs().sum(dim=1)/ms.model.hidden_size
            m_meds = h_meds.abs().sum(dim=1)/ms.model.hidden_size
            m_gmed = h_gmed.abs().sum(dim=1)/ms.model.hidden_size
            m_tub = torch.cat([a for a in l_tub], dim=1).abs().mean(dim=1)
            m_tubles = torch.cat([a for a in l_tubles], dim=1).abs().mean(dim=1)

            # Calculate the mean
            mu_notn.append(m_notn.mean().cpu().item())
            mu_meds.append(m_meds.mean().cpu().item())
            mu_gmed.append(m_gmed.mean().cpu().item())
            mu_tub.append(m_tub.mean().cpu().item())
            mu_tubles.append(m_tubles.mean().cpu().item())

            # Calculate the standard error of the mean
            sig_notn.append(m_notn.std().cpu().item()/np.sqrt(batch_size))
            sig_meds.append(m_meds.std().cpu().item()/np.sqrt(batch_size))
            sig_gmed.append(m_gmed.std().cpu().item()/np.sqrt(batch_size))
            sig_tub.append(m_tub.std().cpu().item()/np.sqrt(batch_size))
            sig_tubles.append(m_tubles.std().cpu().item()/np.sqrt(batch_size))

            # Perform welch's t-test on relevant adjacent pairs
            p_values.append([
                _t_test(m_gmed, m_tubles),
                _t_test(m_tubles, m_tub),
                _t_test(m_tub, m_meds)
            ])

        fig, axes = plt.subplots(1, figsize=(14,10))

        x = np.arange(1,data.shape[0]+1)

        axes.errorbar(x, mu_notn, yerr=sig_notn, capsize=4, label="input")
        axes.errorbar(x, mu_meds, yerr=sig_meds, capsize=4, label="cat. median")
        axes.errorbar(x, mu_gmed, yerr=sig_gmed, capsize=4, label="median")
        axes.errorbar(x, mu_tub, yerr=sig_tub, capsize=4, label="Bathtub")
        axes.errorbar(x, mu_tubles, yerr=sig_tubles, capsize=4, label="lesioned")

        # Plot asterisks for statistical significance between adjacent pairs of means
        def sig_asterix(m1, m2, p):
            y = (m1 - m2) / 2 + m2
            if p < 0.001:
                axes.text(i+1, y-0.0005, r'***', fontsize=16)
            elif p < 0.01:
                axes.text(i+1, y-0.0005, r'**', fontsize=16)
            elif p < 0.05:
                axes.text(i+1, y-0.0005, r'*', fontsize=16)
            else:
                axes.text(i+1, y-0.0005, r'$-$', fontsize=16)
        for i in range(1,9):
            sig_asterix(mu_gmed[i], mu_tubles[i], p_values[i][0])
            sig_asterix(mu_tubles[i], mu_tub[i], p_values[i][1])
            sig_asterix(mu_tub[i], mu_meds[i], p_values[i][2])

        axes.xaxis.set_major_locator(MaxNLocator(integer=True));
        axes.set_xlabel(r"time $\longrightarrow$",fontsize=16)
        axes.set_ylabel("Mean absolute activity",fontsize=16)
        axes.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.0, 0.53))
        axes.grid()
        fig.tight_layout()

        if save is True:
            save_fig(fig, ms.title+"/activity-vs-lesioned", bbox_inches='tight')

#
# Appendix
#

def weights_mean_activity(ms:ModelState, dataset:Dataset, save=True):
    def Whmean(steps):
        # calculates mean network activity after certain amount of timesteps
        # and times this by the weight matrix to get the mean presynaptic activity of the recurrent connections
        b, _ = dataset.create_batches(batch_size=-1, sequence_length=steps, shuffle=True)
        batch = b.squeeze(0)

        h = ms.model.init_state(1)
        for i in range(steps):
            h, l_a = ms.model(batch[i], state=h)

        return (h.mean(dim=0) * ms.model.W.t()).t().detach().cpu()

    # Weights times mean activity at timestep 2
    fig, axes = unit_projection(Whmean(2), colorbar=False, axes_visible=False);
    if save is True:
        save_fig(fig, ms.title+"/weights_mean_activity_t2", bbox_inches='tight')

    # Weights times mean activity at timestep 9
    fig,axes = unit_projection(Whmean(9), colorbar=False, axes_visible=False);
    if save is True:
        save_fig(fig, ms.title+"/weights_mean_activity_t9", bbox_inches='tight')


def pred_after_timestep(ms:ModelState, dataset:Dataset, mask=None, save=True):
    imgs=[]
    for d in range(10):
        imgs = imgs + [ms.predict(_run_seq_from_digit(d, i, ms, dataset, mask=mask)).mean(dim=0) for i in range(1,9)]

    fig, axes = display(imgs, shape=(8,10), axes_visible=False)

    for i in range(10):
        axes[i][0].set_ylabel(str(i)+":", ha='center', fontsize=30, rotation=0, labelpad=30)
    for i in range(8):
        axes[9][i].set_xlabel(str(i+2), ha='center', fontsize=30, rotation=0, labelpad=30)
    fig.text(0.5, -0.015, r"timestep (t)", ha='center', fontsize=30);

    fig.tight_layout()

    if save:
        save_fig(fig,
                      ms.title+"/pred-after-timestep" + ("-lesioned" if mask is not None else ""),
                      bbox_inches='tight')
