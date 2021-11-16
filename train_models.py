import torch
import functions
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

INPUT_SIZE_MNIST = 28*28
INPUT_SIZE_CIFAR = 32*32*3
BATCH_SIZE = 32
SEQ_LENGTH = 10
LOSS_FN = functions.L1Loss
# dataset loaders
import mnist
import cifar

import Network

from train import train

training_set_m, validation_set_m, test_set_m = mnist.load(val_ratio=0.0)
training_set_c, validation_set_c, test_set_c = cifar.load(val_ratio=0.0, color=True)
"""
Create and train ten instances of energy efficient RNNs for MNIST & CIFAR10
"""
N = 10 # number of model instances

# train MNIST networks
for i in range(N):
        mnist_net= Network.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            input_size=INPUT_SIZE_MNIST,
            hidden_size=INPUT_SIZE_MNIST,
            title="/networks/mnist_net",
            device=DEVICE)
    
   
    
        train(mnist_net,
              train_ds=training_set_m,
              test_ds=test_set_m,
              loss_fn=LOSS_FN,
              num_epochs=200,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False)
        
       
    
        # # save model
        mnist_net.save()

# train cifar networks
for i in range(N):        
    cifar_net= Network.State(activation_func=torch.nn.ReLU(),
    optimizer=torch.optim.Adam,
    lr=1e-4,
    input_size=INPUT_SIZE_CIFAR,
    hidden_size=INPUT_SIZE_CIFAR,
    title="/networks/cifar_nets/cifar_net",
    prevbatch=True,
    device=DEVICE)
    
       
    
    train(cifar_net,
      train_ds=training_set_c,
      test_ds=test_set_c,
      loss_fn=LOSS_FN,
      num_epochs=1000,
      batch_size=BATCH_SIZE,
      sequence_length=SEQ_LENGTH,
      verbose=False)
    
    
    ## save model
    cifar_net.save()
