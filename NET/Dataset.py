from __future__ import print_function
import torch
from torch import utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
from parser import *


class Dataset_renju(Dataset):
    def __init__(self, tX, tY=None, transforms=None, train=True):
        self.X = tX
        self.transforms = transforms
        self.train = train

        if train:
            self.Y = tY
        else:
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        elem = self.X[item]
        if self.transforms is not None:
            elem = self.transforms(self.X[item])
        if self.train:
            return elem, self.Y[item]
        else:
            return elem


def make_dataset(count1, count2):
    x, y = parse(count1, count2)
    dataset = Dataset_renju(x, y)
    return dataset
