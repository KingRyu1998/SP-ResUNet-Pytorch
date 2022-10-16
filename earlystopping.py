# -*- coding: utf-8 -*-
'''
@Time    : 2022/9/25 8:36
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : Sprog_CBAM_Unet
@File    : earlystopping.py
@Language: Python3
'''

import torch
from torch import nn
import numpy as np

class EarlyStopping:

    def __init__(self, patience=7, verbose=True):
        self.earlystopping = False
        self.patience = patience
        self.counter = 0
        self.min_val_loss = np.inf

    def __call__(self, val_loss, model_dict, save_path, epoch, local_rank):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if local_rank == 0:
                self.save_checkpoint(val_loss, model_dict, epoch, save_path)

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(
            model, save_path + "/" +
                   "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        self.val_loss_min = val_loss
