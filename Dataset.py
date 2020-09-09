import torch
from typing import List
from types import SimpleNamespace

# Named tuple to store dataset properties
class Dataset(SimpleNamespace):
    x: torch.FloatTensor
    y: torch.FloatTensor
    indices: List[torch.FloatTensor]