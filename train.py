import torch
import numpy as np
from typing import Callable
from functions import *
from ModelState import ModelState
from Dataset import Dataset

def test_epoch(ms: ModelState,
               dataset: Dataset,
               loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
               sequence_length: int):
    data, labels = dataset.create_batches(batch_size=-1, sequence_length=sequence_length, shuffle=False)
    batch_size = data.shape[2]

    with torch.no_grad():
        loss, res = ms.run(data[0], loss_fn)

        print("Test loss:     {:.8f}".format(loss.item()))

        return loss.item(), res

def train_batch(ms: ModelState,
                batch: torch.FloatTensor,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]) -> float:

    loss, res = ms.run(batch, loss_fn)
    ms.step(loss)
    ms.zero_grad()
    return loss.item(), res

def train_epoch(ms: ModelState,
                dataset: Dataset,
                loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
                batch_size: int,
                sequence_length: int,
                verbose = True) -> float:

    batches, labels = dataset.create_batches(batch_size=batch_size, sequence_length=sequence_length, shuffle=True)
    num_batches = batches.shape[0]
    batch_size = batches.shape[2]

    t = Timer()
    tot_loss = 0.
    tot_res = None

    for i, batch in enumerate(batches):

        loss, res = train_batch(ms, batch, loss_fn)
        tot_loss += loss

        if tot_res is None:
            tot_res = res
        else:
            tot_res += res

        if verbose and (i+1) % int(num_batches/10) == 0:
            dt = t.get(); t.lap()
            print("Batch {}/{}, ms/batch: {}, loss: {:.5f}".format(i, num_batches, dt / (num_batches/10), tot_loss/(i)))

    tot_loss /= num_batches
    tot_res /= num_batches

    print("Training loss: {:.8f}".format(tot_loss))

    return tot_loss, tot_res

def train(ms: ModelState,
          train_ds: Dataset,
          test_ds: Dataset,
          loss_fn: Callable[[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor],
          num_epochs: int = 1,
          batch_size: int = 32,
          sequence_length: int = 3,
          verbose = False):

    for epoch in range(ms.epochs+1, ms.epochs+1 + num_epochs):
        print("Epoch {}".format(epoch))

        train_loss, train_res = train_epoch(ms, train_ds, loss_fn, batch_size, sequence_length, verbose=verbose)

        test_loss, test_res = test_epoch(ms, test_ds, loss_fn, sequence_length)

        ms.on_results(epoch, train_res, test_res)