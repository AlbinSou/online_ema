#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import TensorDataset, IterableDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from itertools import cycle
import time

class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.transform = transforms.ToTensor()
        self.data = data
        self.sampler = self._sampler()

    def __iter__(self):
        while True:
            i = next(self.sampler)
            yield self.data[i]

    def _sampler(self):
        while True:
            yield np.random.randint(0, len(self.data))

    def append(self, data):
        self.data = np.concatenate((self.data, data))


class MyNormalDataset(Dataset):
    def __init__(self, data):
        self.transform = transforms.ToTensor()
        self.data = data

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def change(self, data):
        self.data = data

dataset = MyNormalDataset(np.arange(100))

loader = DataLoader(dataset, batch_size=1, num_workers=0)
