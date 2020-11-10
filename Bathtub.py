import torch
import torch.nn.functional as F
import numpy as np
from ModelState import ModelState
from functions import *

class Bathtub(torch.nn.Module):
    def __init__(self, beta:float, input_size: int, hidden_size: int, activation_func, weights_init=init_params, term='a'):
        super(Bathtub, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.beta = beta
        self.activation_func = activation_func
        self.term = term
        self.W = torch.nn.Parameter(weights_init(hidden_size, hidden_size))

    def forward(self, x, state=None, synap_trans=False):
        if state is None:
            state = self.init_state(x.shape[0])
        h = state
        h_past = h
        # pad input so it matches the hidden state dimensions
        x_pad = F.pad(x, (0, self.hidden_size-self.input_size), "constant", 0)
        a = h @ self.W + x_pad
        h = self.activation_func(a)
        if self.term == 'a': #beta*|W| * |h_t-1| + |h_t|
          
            l_a = [self.beta *(torch.abs(h_past) @ torch.abs(self.W)), a, h] # MEA
        elif self.term == 'b': #beta*|W| + |h_t|
          
            l_a = [self.beta * torch.abs(self.W), a, (1 - self.beta)*h] # MEA
        else: # term c: beta*|W| + |a_t|
           
            l_a = [self.beta * torch.abs(self.W), a, (1 - self.beta)*a] # MEA
        
        # return state vectors, activity vectors (on which the loss function applies)
        if synap_trans: # use synaptic transmission instead
            synaptrans = (torch.abs(h_past) @ torch.abs(self.W))
            #l_a = [self.beta *(torch.abs(h_past) @ torch.abs(self.W)), synaptrans, h]
            l_a = [l_a[0]] + [synaptrans] + [l_a[-1]]
        return h, l_a

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))


class State(ModelState):
    def __init__(self,
                 activation_func,
                 optimizer,
                 lr:float,
                 beta:float,
                 title:str,
                 input_size:int,
                 hidden_size:int,
                 device:str,
                 deterministic=True,
                 weights_init=init_params,
                 term='a',
                 seed=0):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        ModelState.__init__(self,
                            Bathtub(beta,input_size, hidden_size, activation_func, weights_init=weights_init, term=term).to(device),
                            optimizer,
                            lr,
                            
                            title,
                            {
                                "train loss": np.zeros(0),
                                "test loss": np.zeros(0)
                            },
                            device)

    def run(self, batch, loss_fn):
        """
        Runs a batch of sequences through the model

        Returns:
            loss,
            training metadata
        """
        sequence_length = batch.shape[0]
        batch_size = batch.shape[1]

        h = self.model.init_state(batch_size)

        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)

        for i in range(sequence_length):
            h, l_a = self.model(batch[i], state=h) # l_a is now a list of three terms l_a[1] is for comparison
            loss = loss + self.loss(l_a[0], loss_fn) + self.loss(l_a[-1], loss_fn)

        return loss, loss.detach()

    def loss(self, l_a, loss_fn):
        return loss_fn(l_a)

    def predict(self, state):
        """
        Returns the networks 'prediction' for the input.
        """
        pred = state @ self.model.W
        return pred[:,:self.model.input_size]

    def step(self, loss):
        loss.backward()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def on_results(self, epoch:int, train_res, test_res):
        """Save training metadata
        """
        append_dict(self.results, {"train loss": train_res.cpu().numpy(), "test loss": test_res.cpu().numpy()})
        self.epochs += 1
