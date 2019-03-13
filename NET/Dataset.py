from __future__ import print_function
import torch
from torch import utils
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from parser import *
import gc


def make_dataset(count1, count2):
    x, y = parse(count1, count2)
    dataset = TensorDataset(x, y)
    del x, y
    gc.collect()
    return dataset
