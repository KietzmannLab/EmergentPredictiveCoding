import torch
from functions import *
import argparse
parser = argparse.ArgumentParser(description='device')
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
SEQ_LENGTH = 10
LOSS_FN = L1Loss

import mnist

import Network

from train import train

training_set, validation_set, test_set = mnist.load(val_ratio=0.0)

"""
Create and train ten instances of energy efficient RNNs
"""
N = 10 # number of model instances


for i in range(N):
    
        net= Network.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            input_size=INPUT_SIZE,
            hidden_size=INPUT_SIZE,
            title="/networks/preact_net",
            device=DEVICE)
    
   
    
        train(net,
              train_ds=training_set,
              test_ds=test_set,
              loss_fn=LOSS_FN,
              num_epochs=200,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False)
        
       
    
        # # save model
        net.save()
