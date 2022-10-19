#### Imports
import torch
import functions
import mnist
import Network
import plot

import cifar


#### Load datasets
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
# load MNIST
training_set_m, validation_set_m, test_set_m = mnist.load(val_ratio=0.0)
# load CIFAR10
training_set_c, validation_set_c, test_set_c = cifar.load(val_ratio=0.0, color=True)

#### Load trained networks for MNIST & CIFAR10
mnist_nets = []
cifar_nets = []
n_instances = 10
# load networks for bootstrap
for i in range(0, n_instances):
    mnist_net = Network.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            input_size=INPUT_SIZE_MNIST,
            hidden_size=INPUT_SIZE_MNIST,
            title="networks/mnist_networks/mnist_net",
            device=DEVICE)
    mnist_net.load(i)
    mnist_nets.append(mnist_net)
    
    cifar_net = Network.State(activation_func=torch.nn.ReLU(),
            optimizer=torch.optim.Adam,
            lr=1e-4,
            input_size=INPUT_SIZE_CIFAR,
            hidden_size=INPUT_SIZE_CIFAR,
            title="networks/cifar_networks/cifar_net",
            device=DEVICE)
    cifar_net.load(i)
    cifar_nets.append(cifar_net)
    
#### Figure 4B: evolution of preactivation for model, lesioned model, control & three benchmarks¶
plot.bootstrap_model_activity(mnist_nets, training_set_m, test_set_m, seed=None, lesioned=True, save=True, data_type='mnist')

#### Figure 5C: evolution of preactivation for model, lesioned model, control & three benchmarks for CIFAR10
plot.bootstrap_model_activity(cifar_nets, training_set_c, test_set_c, seed=None, lesioned=True, save=True, data_type='cifar')
