import torch
import torch.nn.functional as F
import numpy as np
from ModelState import ModelState
import functions
from torch import nn

class Network(torch.nn.Module):
    """
    Recurrent Neural Network class containing parameters of the network
    and computes the forward pass.
    Returns hidden state of the network and preactivations of the units. 
    """
    def __init__(self, input_size: int, hidden_size: int, activation_func, weights_init=functions.init_params, prevbatch=False, conv=False):
        super(Network, self).__init__()

        self.input_size = input_size
        if conv:
            self.conv = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=3)
        self.is_conv= conv
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.W = torch.nn.Parameter(weights_init(hidden_size, hidden_size))
        self.prevbatch = prevbatch

    def forward(self, x, state=None, synap_trans=False):

        if state is None:
            state = self.init_state(x.shape[0])
        h = state

        # pad input so it matches the hidden state dimensions
        if not self.is_conv:
            x_pad = F.pad(x, (0, self.hidden_size-self.input_size), "constant", 0)
            a = h @ self.W + x_pad

        h = self.activation_func(a)
        # return state vector and preactivation vector 
        return h, [a]

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))


class State(ModelState):
    def __init__(self,
                 activation_func,
                 optimizer,
                 lr:float,
                 title:str,
                 input_size:int,
                 hidden_size:int,
                 device:str,
                 deterministic=True,
                 weights_init=functions.init_params,
                 prevbatch=False,
                 conv=False,
                 seed=0):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
           
        ModelState.__init__(self,
                            
                            Network(input_size, hidden_size, activation_func, weights_init=weights_init, prevbatch=prevbatch, conv=conv).to(device),
                            optimizer,
                            lr,
                            
                            title,
                            {
                                "train loss": np.zeros(0),
                                "test loss": np.zeros(0)
                            },
                            device)

    def run(self, batch, loss_fn, state=None):
        """
        Runs a batch of sequences through the model

        Returns:
            loss,
            training metadata
        """
        sequence_length = batch.shape[0]
        batch_size = batch.shape[1]
        if state == None:
            h = self.model.init_state(batch_size)
        else:
            if self.model.prevbatch:
                
                h = state
            else:
                h = self.model.init_state(batch_size)

        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)

        for i in range(sequence_length):
            h, l_a = self.model(batch[i], state=h) # l_a is now a list of three terms l_a[1] is for comparison
            loss = loss + self.loss(l_a[0], loss_fn) 

        return loss, loss.detach(), state

    def loss(self, l_a, loss_fn):
        return loss_fn(l_a)

    def predict(self, state, latent=False):
        """
        Returns the networks 'prediction' for the input.
        """
        pred = state @ self.model.W
        if not latent:
            return pred[:,:self.model.input_size]
        return pred[:,:]

    def step(self, loss):
        loss.backward()
        #nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def on_results(self, epoch:int, train_res, test_res):
        """Save training metadata
        """
        functions.append_dict(self.results, {"train loss": train_res.cpu().numpy(), "test loss": test_res.cpu().numpy()})
        self.epochs += 1
