import torch
import numpy as np
from ModelState import ModelState
from functions import *

class LSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation_func):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func

        self.representation = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=False)
        self.prediction = torch.nn.Parameter(init_params(input_size, hidden_size))

    def forward(self, x:torch.FloatTensor, state=None):
        if state is None:
            state = self.init_state(x.shape[0])
        pred, lstm_state = state

        a = x + pred
        error = self.activation_func(a)
        lstm_state = self.representation(error, lstm_state)
        pred = lstm_state[0] @ self.prediction.t()

        # return state vectors, activity vectors (on which the loss function applies)
        return (pred, lstm_state), [a, lstm_state[0]]

    def init_state(self, batch_size: int):
        return (torch.zeros(batch_size, self.input_size), (torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size))))


class RNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation_func):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func

        self.W_error = torch.nn.Parameter(init_params(hidden_size, input_size))
        self.W_state = torch.nn.Parameter(init_params(hidden_size, hidden_size))
        self.W_pred = torch.nn.Parameter(init_params(input_size, hidden_size))

    def forward(self, x, state=None):
        if state is None:
            state = self.init_state(x.shape[0])
        pred, h = state

        a = x + pred
        error = self.activation_func(a)

        h_a = error @ self.W_error.t() + h @ self.W_state.t()
        h = self.activation_func(h_a)
        pred = h @ self.W_pred.t()

        # return state vectors, activity vectors (on which the loss function applies)
        return (pred, h), [a, h_a]

    def init_state(self, batch_size: int):
        return (torch.zeros(batch_size, self.input_size), torch.zeros((batch_size, self.hidden_size)))


class State(ModelState):
    def __init__(self,
                 modeltype,
                 activation_func,
                 optimizer,
                 lr:float,
                 title:str,
                 input_size:int,
                 hidden_size:int,
                 device:str,
                 total_loss=True,
                 deterministic=True,
                 seed=0):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        ModelState.__init__(self,
                            modeltype(input_size, hidden_size, activation_func).to(device),
                            optimizer,
                            lr,
                            title,
                            {
                                "train loss": np.zeros(0),
                                "train l1 activations": np.zeros(0),
                                "train l2 activations": np.zeros(0),
                                "test loss": np.zeros(0),
                                "test l1 activations": np.zeros(0),
                                "test l2 activations": np.zeros(0)
                            },
                            device)
        self.total_loss = total_loss

    def run(self, batch, loss_fn):
        """Runs a batch of sequences through the model

        Returns:
            loss,
            training metadata
        """
        sequence_length = batch.shape[0]
        batch_size = batch.shape[1]

        state = self.model.init_state(batch_size)
        loss_l1 = torch.zeros(1, dtype=torch.float, requires_grad=True)
        loss_l2 = torch.zeros(1, dtype=torch.float, requires_grad=True)
        tot_size = self.model.input_size + self.model.hidden_size

        for i in range(sequence_length):
            state, l_a = self.model(batch[i], state=state)

            loss_l1 = loss_l1 + loss_fn(l_a[0]) * (self.model.input_size / tot_size)
            loss_l2 = loss_l2 + loss_fn(l_a[1]) * (self.model.hidden_size / tot_size)

        loss = loss_l1 + loss_l2 if self.total_loss else loss_l1

        return loss, torch.Tensor([loss.item(), loss_l1.item(), loss_l2.item()])

    def predict(self, state):
        """Returns the networks 'prediction' for the input.
        """
        return state[0]

    def loss(self, l_a, loss_fn):
        return loss_fn(torch.cat([a.flatten() for a in l_a]))

    def step(self, loss):
        loss.backward()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def on_results(self, epoch:int, train_res, test_res):
        """Save training metadata
        """
        append_dict(self.results,
            {
                "train loss": np.array([train_res[0].item()]),
                "train l1 activations": np.array([train_res[1].item()]),
                "train l2 activations": np.array([train_res[2].item()]),
                "test loss": np.array([test_res[0].item()]),
                "test l1 activations": np.array([test_res[1].item()]),
                "test l2 activations": np.array([test_res[2].item()])
            })
        self.epochs += 1