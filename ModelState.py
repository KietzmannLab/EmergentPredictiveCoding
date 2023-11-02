import torch
import os
import re

class ModelState:
    """A class to encapsulate a neural network model with a number of attributes associated with it.

    Serves as a place to store associated attributes to a model, such as the optimizer or training metadata.
    """
    def __init__(self,
                 model,
                 optimizer,
                 lr:float,
                 title:str,
                 results,
                 device:str):

        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.title = title
        self.epochs = 0
        self.results = results
        self.device = device

    def save(self):
        filepath = "./models/" + self.title +".pth"

        torch.save({
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "results": self.results
        }, filepath)

    def load(self, idx=None):
        if (idx is None):
            filepath = "./models/" + self.title +".pth"
        else:
            filepath = "./models/" + self.title +"_" + str(idx) + ".pth"

        state = torch.load(filepath, map_location=torch.device(self.device))
        self.epochs = state['epochs']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.results = state['results']
