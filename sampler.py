import pandas as pd
import numpy as np
import torch
import io, pickle, os,random
from torch.utils.data import Dataset as dataset

class Sampler(dataset):
    """Partition sampler by batch."""
    def __init__(self, X, Y, nbatch = 16):
        self.sq_list = X
        self.gt_list = np.squeeze(Y)
        self.batch_size = nbatch

    def __getitem__(self, index):
        # 1. get sequnece and its GT
        index = index % len(self.sq_list)
        sq = self.sq_list[index]
        gt = [self.gt_list[index]]
        # 2. transform to tensor
        sq = torch.FloatTensor(sq)
        gt = torch.LongTensor(gt).squeeze(0)
        return sq, gt
    
    def __len__(self):
        return len(self.gt_list)
    
