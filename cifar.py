import torch
import torchvision
from torchvision import transforms
from Dataset import Dataset
import random

class CIFAR10Dataset(Dataset):
    """Container class for the CIFAR10 database containing Tensors with the images and labels, as well as a list of indices for each category
    """
    def __init__(self, x, y, indices, repeat=1):
        super(Dataset, self).__init__(x=x, y=y, indices=indices)
        self.repeat = repeat


    def create_batches(self, batch_size, sequence_length, shuffle=True,  distractor=False, fixed_starting_point=None):
        data, labels = create_sequences(self, sequence_length, batch_size, shuffle, distractor, fixed_starting_point)
        data = data.repeat_interleave(self.repeat, dim=1)
        labels = labels.repeat_interleave(self.repeat, dim=1)
        return data, labels


def create_sequences(dataset, sequence_length, batch_size, shuffle=True, distractor=False, fixed_starting_point=None):
    # number of datapoints
    data_size, ninputs = dataset.x.shape

    # maximum theoretical amount of sequences
    max_sequences = int(data_size / sequence_length)

    # for test and validation it is not actually necessary to shuffle,
    # so for consistent testing/validation we can use the same sequences every time
    if shuffle:
        # shuffle all the data points per digit class
        indices = [dataset.indices[i][torch.randperm(d.shape[0])] for i,d in enumerate(dataset.indices)]
        # choose random sequence starting points
        seq_starting_points = torch.randperm(max_sequences)
    else:
        indices = dataset.indices
        seq_starting_points = torch.arange(max_sequences)
    # if we want the same starting digit for all the sequences
    if fixed_starting_point is not None:
        assert(isinstance(fixed_starting_point, int) and fixed_starting_point in list(range(10)))
        seq_starting_points = torch.ones(max_sequences) * fixed_starting_point
    # from the starting points, create sequences of the required length
    # first we repeat each starting point 'sequence_length' times
    sequences = seq_starting_points.repeat_interleave(sequence_length).view(max_sequences, sequence_length)
    # we then add to each digit the index of its position within the sequence,
    # so we get increasing numbers in the sequence
    for i in range(1, sequence_length):
        sequences[:,i] += i
    # take the remainder of all numbers in sequence to get actual digits from 0-9
    sequences %= 10

    # switch out digit at position 8 for a distractor if flag is true
    if distractor:
        for i in range(max_sequences):
            digit = sequences[i,8]
            candidates = list(range(0,10))
            candidates.remove(digit)
            sequences[i, 8] = random.choice(candidates)

    # flatten again
    sequences = sequences.flatten()
    # create an array to store the indices for the digits in 'data'
    epoch_indices = torch.zeros(data_size, dtype=torch.long)
    # because not every digit is equally represented,
    # we have to keep track of where in the sequence we have run out of
    # digits. This 'cutoff' is the minimum between all digits
    cutoff = data_size

    for i in range(10):
        # mask to filter out the positions of this digit
        mask = sequences==i
        # calculating the cumulative sum of the mask gives us a nice increasing
        # index exactly at the points of where the digit is in the list of sequences.
        # we can use this as an index for 'indices'
        indices_idx = torch.cumsum(mask, 0)
        # we cut 'idx' off where the index exceeds the number of digits we actually have
        # for this case
        indices_idx = indices_idx[indices_idx < indices[i].shape[0]]
        # keep track of the earliest cutoff point for later
        cutoff = min(cutoff, indices_idx.shape[0])
        # also cutoff the mask so it has the right shape
        mask = mask[:indices_idx.shape[0]]
        # we select the data indices from 'indices' with 'indices_idx', mask that
        # so we are left with the data indices on the positions where the digits occur
        # in the sequences
        epoch_indices[:indices_idx.shape[0]][mask] = indices[i][indices_idx][mask]

    # if batch_size is invalid, create one big batch
    if batch_size < 1 or batch_size > int(cutoff / sequence_length):
        batch_size = int(cutoff / sequence_length)

    # we cut off the cutoff point so we can create an integer amount of batches and sequences
    cutoff = cutoff - cutoff % (batch_size * sequence_length)

    epoch_indices = epoch_indices[:cutoff]
    sequences = sequences[:cutoff]
    # select the data points and group per sequence and batch
    #x = dataset.x[epoch_indices].view(-1, batch_size, sequence_length, 32*32).transpose(1,2)
    x = dataset.x[epoch_indices].view(-1, batch_size, sequence_length, ninputs).transpose(1,2)
    y = sequences.view(-1, batch_size, sequence_length).transpose(1,2)
    return x, y


def load(val_ratio = 0.1, color=False):
    """Load CIFAR10 data, transform to tensors and grayscale (if color=True) and calculate indices for each category
    """
    if color:
        transform = transforms.ToTensor()
    
        nchannels = 3
        
    else: # gray scale
        transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()
       ])
        nchannels = 1
        
    train_data = torchvision.datasets.CIFAR10("./datasets/", train=True, transform=transform, download=True)
    test_data  = torchvision.datasets.CIFAR10("./datasets/", train=False, transform=transform, download=True)

    validation_size = int(val_ratio * len(train_data))
    train_size = len(train_data) - validation_size
    ninputs = 32,32
    train_dimens = (train_size, nchannels, ninputs[0]*ninputs[1])
    val_dimens = (validation_size, nchannels, ninputs[0]*ninputs[1])

    train_x = torch.zeros(train_dimens)
    train_y = torch.zeros(train_size, dtype=torch.int)

    val_x = torch.zeros(val_dimens)
    val_y = torch.zeros(validation_size, dtype=torch.int)
    for i, d in enumerate(train_data):
        if i < train_size:
            train_x[i] = d[0].view(train_dimens[1], train_dimens[-1])
            train_y[i] = d[1]
        else:

            val_x[i-train_size] = d[0].view(val_dimens[0], val_dimens[-1])
            val_y[i-train_size] = d[1]
    # safe image indices for each category
    train_indices = [torch.nonzero(train_y==i).flatten() for i in range(10)]
    val_indices = [torch.nonzero(val_y==i).flatten() for i in range(10)]
    training_set = CIFAR10Dataset(x=train_x, y=train_y, indices=train_indices)
    validation_set = CIFAR10Dataset(x=val_x, y=val_y, indices=val_indices)
    
    test_size = len(test_data)
    test_dimens = (test_size, nchannels, ninputs[0]*ninputs[1])

    test_x = torch.zeros(test_dimens)
    test_y = torch.zeros(test_size, dtype=torch.int)
    for i, d in enumerate(test_data):
        test_x[i] = d[0].view(test_dimens[1], test_dimens[-1])
        test_y[i] = d[1]
    test_indices = [torch.nonzero(test_y==i).flatten() for i in range(10)]
    test_set = CIFAR10Dataset(x=test_x, y=test_y, indices=test_indices)
    
    # reshape to do a horizontal stack of color channels
    training_set.x = training_set.x.view(train_dimens[0], train_dimens[1]*train_dimens[2])
    validation_set.x = validation_set.x.view(val_dimens[0], val_dimens[1]*val_dimens[2])
    test_set.x = test_set.x.view(test_dimens[0], test_dimens[1]*test_dimens[2])
    return training_set, validation_set, test_set

def means(dataset:CIFAR10Dataset):
    means = torch.Tensor(10,32*32)
    for i in range(10):
        means[i] = torch.mean(dataset.x[dataset.indices[i]],dim=0)
    return means

def medians(dataset:CIFAR10Dataset):
    ndata, ninputs = dataset.x.shape
    medians = torch.Tensor(10,ninputs)
    for i in range(10):
            medians[i] = torch.median(dataset.x[dataset.indices[i]],dim=0).values
    return medians

if __name__ == '__main__':
   train, val, test = load()
