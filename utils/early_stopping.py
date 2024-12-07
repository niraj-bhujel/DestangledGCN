#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:37:55 2020

@author: dl-asoro

https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, metric_name = 'val_loss', last_epoch=0,
                 ckpt_path='checkpoint.pt', trace_func=print, verbose=True, debug=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            ckpt_path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_val_min = np.Inf
        self.delta = delta
        self.ckpt_path = ckpt_path
        self.trace_func = trace_func
        self.metric_name = metric_name
        self.debug = debug
        self.last_epoch = last_epoch
        
    def __call__(self, metric_val, model, epoch=None):

        score = -metric_val

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_val, model)

        elif score < self.best_score + self.delta:
            if epoch is None:
                self.counter += 1
            else:
                self.counter = (epoch-self.last_epoch)

            self.trace_func("{} didn't improve from {:.6f}. EarlyStopping counter: {}/{}".format(
                self.metric_name, self.metric_val_min, self.counter, self.patience))
            if self.counter>self.patience:
                self.early_stop = True

        else:

            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch

            self.best_score = score
            self.save_checkpoint(metric_val, model)
            self.counter = 0
            

    def save_checkpoint(self, metric_val, model):
        '''Saves model when validation loss decrease.'''
        self.trace_func("{} improved from {:.6f} to {:.6f}.".format(
            self.metric_name, self.metric_val_min, metric_val))
        self.metric_val_min = metric_val
        if not self.debug:
            torch.save(model.state_dict(), self.ckpt_path + '/min_{}.pth'.format(self.metric_name))
        