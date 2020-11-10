import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Optional
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from functions import *
import argparse
parser = argparse.ArgumentParser(description='device')
#parser.add_argument('--disable-cuda', action='store_true', help='Disable Cuda')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    DEVICE = 'cpu'

print('Using {}'.format(DEVICE))

INPUT_SIZE = 28*28
BATCH_SIZE = 32
SEQ_LENGTH = 9
LOSS_FN = L1Loss

#betas = list(np.logspace(-10, -1, 10)) + list(np.linspace(0, 1, 11).round(1)) # both log and linear scale
betas = list(np.linspace(0.997, 0.998, 11).round(4))
import mnist
import plot
import Bathtub
import PretNet
from train import train

training_set, validation_set, test_set = mnist.load(val_ratio=0.0)

HIDDEN_SIZE = 64
terms = ['b']
for term in terms:
    for beta in betas:
        # lstm = PretNet.State(
        #     modeltype=PretNet.LSTM,
        #     activation_func=torch.nn.ReLU(),
        #     optimizer=torch.optim.Adam,
        #     lr=5e-4,
        #     input_size=INPUT_SIZE,
        #     hidden_size=HIDDEN_SIZE,
        #     title="pretnet/lstm-relu-seq9",
        #     device=DEVICE)
        
        # rnn = PretNet.State(
        #     modeltype=PretNet.RNN,
        #     activation_func=torch.nn.ReLU(),
        #     optimizer=torch.optim.Adam,
        #     lr=5e-4,
        #     input_size=INPUT_SIZE,
        #     hidden_size=HIDDEN_SIZE,
        #     title="pretnet/rnn-relu-seq9",
        #     device=DEVICE)
        
        tub = Bathtub.State(
            activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            beta=beta,
            input_size=INPUT_SIZE,
            hidden_size=INPUT_SIZE + HIDDEN_SIZE,
            title="bathtub/64u-relu-weight-act-l1-"+str(term)+'-'+str(beta),
            term=term,
            device=DEVICE)
        
        ms = Bathtub.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            beta=beta,
            input_size=INPUT_SIZE,
            hidden_size=INPUT_SIZE,
            title="bathtub/relu-weight-act-l1-"+str(term)+'-'+str(beta),
            term=term,
            device=DEVICE)
        
        # train(lstm,
        #       train_ds=training_set,
        #       test_ds=test_set,
        #       loss_fn=LOSS_FN,
        #       num_epochs=200,
        #       batch_size=BATCH_SIZE,
        #       sequence_length=SEQ_LENGTH,
        #       verbose=False)
        
        # train(rnn,
        #       train_ds=training_set,
        #       test_ds=test_set,
        #       loss_fn=LOSS_FN,
        #       num_epochs=200,
        #       batch_size=BATCH_SIZE,
        #       sequence_length=SEQ_LENGTH,
        #       verbose=False)
        
        train(tub,
              train_ds=training_set,
              test_ds=test_set,
              loss_fn=LOSS_FN,
              num_epochs=200,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False)
        
        train(ms,
              train_ds=training_set,
              test_ds=test_set,
              loss_fn=LOSS_FN,
              num_epochs=200,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False)
        
        # # save models
        # lstm.save()
        # rnn.save()
        tub.save()
        ms.save()