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
        filepath = "./models/" + self.title
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        files = []
        for file in os.listdir(os.path.dirname(filepath)):
            if re.search("^" + os.path.basename(filepath) + "-[0-9]+\.pth$", file):
                files.append(file)
        idx = len(files)
        filepath = filepath + "-" + str(idx) + ".pth"

        torch.save({
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "results": self.results
        }, filepath)

    def load(self, idx=None):
        filepath = "./models/" + self.title
        if idx is None:
            files = []
            for file in os.listdir(os.path.dirname(filepath)):
                if re.search("^" + os.path.basename(filepath) + "-[0-9]+\.pth$", file):
                    files.append(file)
            idx = len(files) - 1

        filepath = "./models/" + self.title + "-" + str(idx) + ".pth"

        state = torch.load(filepath, map_location=torch.device(self.device))
        self.epochs = state['epochs']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.results = state['results']
