# -*- coding: utf-8 -*-
'''
@Time    : 2022/9/18 15:44
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : Sprog_CBAM_Unet
@File    : data_iter.py
@Language: Python3
'''

import os
import numpy as np
import torch
from torch import nn
from preprocessing.bandpass_filters import filter_gaussian
from preprocessing.decomposition import decomposition_fft

class DataIter(nn.Module):

    def __init__(self, dir_path, n_cascade):
        super().__init__()
        self.dir_path = dir_path
        self.n_cascade = n_cascade

    def __getitem__(self, idx):
        sample_list = os.listdir(self.dir_path)
        sample_name = sample_list[idx]
        sample_path = os.path.join(self.dir_path, sample_name)
        inputs, labels = self.read_data(sample_path)
        decomp_inputs = self.filed_decomposition(inputs, self.n_cascade)
        decomp_inputs = torch.from_numpy(decomp_inputs)
        labels = torch.from_numpy(labels)
        return decomp_inputs, labels

    @staticmethod
    def read_data(data_path):
        data = np.load(data_path)
        input = torch.tensor(data[:6])
        label = torch.tensor(data[6:])
        return input, label

    @staticmethod
    def filed_decomposition(field, n_cascade):
        shape = field.shape
        filter = filter_gaussian(shape, n_cascade)
        decomp = decomposition_fft(field, filter, compute_stats=True)
        for i in range(n_cascade):
            mu = decomp['means'][i]
            sigma = decomp['stds'][i]
            decomp['cascade_levels'][i] = (decomp['cascade_levels'][i] - mu) / sigma
        return decomp['cascade_levels']
