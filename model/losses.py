#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:39:16 2021

@author: dl-asoro
"""

import torch
import numpy as np
import torch.distributions as tdist
import time
import pickle
import torch.nn as nn

def constrained_loss(y_true, y_pred, lower_b=0., upper_b=2.0):
    too_big = (y_pred > upper_b).type(torch.float64)
    too_small = (y_pred < lower_b).type(torch.float64)
    loss = too_big * torch.abs(y_pred - upper_b) + too_small * torch.abs(y_pred - lower_b)
    return loss.sum(dim=-1)

def l1_loss(y_true, y_pred):
    loss = torch.abs(y_true-y_pred)
    return loss.sum(dim=-1)

def l2_loss(y_true, y_pred):
    loss = (y_true - y_pred)**2
    return loss.sum(dim=-1)

def collision_loss(y_pred, margin=0.5):
    loss = torch.max(torch.zeros_like(y_pred), margin-y_pred)**2
    return loss.sum(dim=-1)

def smoothl1_loss(y_true, y_pred, beta=1,):
    diff = torch.abs(y_true - y_pred)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.sum(dim=-1)

def min_ade_nodes(y_true, y_pred, weights=1.):
    '''
    y_true: (num_nodes, 2)
    y_pred: (K, num_nodes, 2)
    weights: (num_nodes,)
    '''
    err =  y_true.unsqueeze(0)[..., :2] - y_pred[..., :2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=-1)
    err = torch.pow(err, exponent=0.5)
    err, indices = torch.min(err, dim=0) # err-> (num_nodes,) indices -> (num_nodes)

    return torch.mean(err*weights), indices


def min_fde_nodes(y_true, y_pred, final_mask, weights=1.):
    '''
    y_true: (num_nodes, 2)
    y_pred: (K, num_nodes, 2)
    weights: (num_nodes,)
    '''
    err =  y_true.unsqueeze(0)[..., :2] - y_pred[..., :2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=-1)
    err = torch.pow(err, exponent=0.5)
    
    err[..., final_mask==0] = 1e-6 
    err, indices = torch.min(err, dim=0) # err-> (num_nodes,) indices -> (num_nodes)

    return torch.mean(err*weights), indices


def bivariate_loss(y_true, y_pred):
    '''
    Parameters
    ----------
    y_pred : [num_samples, num_nodes, node_dim]
    y_true : [num_samples, num_nodes, node_dim]
    '''

    normx = y_true[..., 0] - y_pred[..., 0]
    normy = y_true[..., 1] - y_pred[..., 1]

    sx = torch.exp(y_pred[..., 2]) #sx
    sy = torch.exp(y_pred[..., 3]) #sy

    corr = torch.tanh(y_pred[..., 4])

    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    numer = torch.exp(-z/(2*negRho))
    
    # Normalization factor
    denom = 2 * np.pi * sxsy * torch.sqrt(negRho)

    # Final PDF calculation
    result = numer / denom

    result = -torch.log(torch.clamp(result, min=1e-12))
    
    return result

    
def bivariate_sample(y_pred):
    '''
    Parameters
    ----------
    y_pred : [num_samples, num_nodes, node_dim], this method is slower
    '''
    start = time.time()

    mean = y_pred[..., 0:2]
    #bi-variate parameters
    sx = torch.exp(y_pred[..., 2]) #sx
    sy = torch.exp(y_pred[..., 3]) #sy
    corr = torch.tanh(y_pred[..., 4])  #corr
    
    # cov = torch.cuda.DoubleTensor(y_pred.size()[:-1] + (2, 2)).fill_(0).to(y_pred.device) #[num_samples*num_nodes, 2, 2]
    # cov[..., 0, 0]= sx*sx
    # cov[..., 0, 1]= corr*sx*sy
    # cov[..., 1, 0]= corr*sx*sy
    # cov[..., 1, 1]= sy*sy
    
    cov = torch.stack([sx*sx, corr*sx*sy,  corr*sx*sy, sy*sy], dim=-1).view(y_pred.size()[:-1] + (2, 2)) + 1e-6
    
    try:    
        scale_tril = torch.linalg.cholesky(cov)
        dist = tdist.MultivariateNormal(mean, scale_tril=scale_tril)
    except Exception as e:
        print(cov)
        raise ValueError('%s'%e)


    # print('time required:', time.time() - start)
    # dist = tdist.MultivariateNormal(mean, cov)
    # dist = tdist.MultivariateNormal(mean, scale_tril=scale_tril)
    # print('time required (after dist) :', time.time() - start)

    return dist

def mv_normal(y_pred, pred_dim=2):
    '''
    https://discuss.pytorch.org/t/learn-a-normal-distribution-with-multivariatenormal-loc-covariance-matrix/55237/3
    '''
    
    loc, tril, diag = y_pred.split(pred_dim, dim=-1)
    
    diag = 1 + torch.nn.functional.elu(diag) + 1e-6
    # diag = diag.clamp(1e-9)

    z = torch.zeros(size=y_pred.size()[:-1], device=y_pred.device)
    # scale_tril is a tensor of size [batch, 2, 2]
    scale_tril = torch.stack([
                             diag[..., 0],   z, 
                             tril[..., 0], diag[..., 1],
                             ], dim=-1).view(y_pred.size()[:-1] + (pred_dim, pred_dim))

    assert not torch.isnan(loc).all(), f'nan values in loc:{loc}'
    assert not torch.isnan(scale_tril).all(), f'nan values in scale_tril:{scale_tril}'

    try:
        dist = tdist.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    except Exception as e:
        # print(y_pred)
        raise ValueError('%s'%e)
    
    # print(f"{dist.batch_shape=}")

    return dist

def mm_likelihood(y_pred, y_true=None):
    '''
    https://github.com/govvijaycal/confidence_aware_predictions/blob/main/scripts/models/multipath.py
    '''
    if y_true is not None:
        dx = y_true[..., 0] - y_pred[..., 0]
        dy = y_true[..., 1] - y_pred[..., 1]
    else:
        dx = y_pred[..., 0]
        dy = y_pred[..., 1]

    log_std1 = torch.clamp(y_pred[..., 2], 0.0, 5.0)
    log_std2 = torch.clamp(y_pred[..., 3], 0.0, 5.0)

    std1 = torch.exp(log_std1)
    std2 = torch.exp(log_std2)

    cos_th = torch.cos(y_pred[..., 4])
    sin_th = torch.sin(y_pred[..., 4])

    reg_log_cov_loss = log_std1 + log_std2
    reg_mahalanobis_loss = 0.5 * (torch.square(dx*cos_th + dy*sin_th)/std1**2 + torch.square(-dx*sin_th + dy*cos_th)/std2**2)

    # probs = torch.nn.functional.softmax(y_pred[..., 5], dim=0) 
    # class_loss = -torch.log(probs)

    loss = reg_log_cov_loss + reg_mahalanobis_loss

    return loss

def sample_mm_likelihood():
    pass


def normal(y_pred, pred_dim=2):
    '''
    y_pred: predicted mean and std (num_samples, num_nodes, 2*pred_dim)
    '''
    mus, log_sigmas = y_pred.split(pred_dim, dim=-1)
    log_simgas = torch.exp(log_sigmas)

    dist = tdist.Normal(mus, log_simgas)
    # loss = -d.log_prob(y_true).sum()

    return dist

def compute_loss(y_true, y_pred, loss_func, weight=1., mode='average'):
    '''
    y_pred: predicted mean and std (num_samples, (num_peds * Tp), pred_dim)
    '''
    K, num_nodes, dim = y_pred.shape
    # assert len(y_pred.shape)==3, f'y_pred shape {y_pred.shape} should be (K, N, dim)'

    if loss_func=='collision_loss':
        loss = collision_loss(y_pred, margin=0.4)

    elif loss_func =='mv_normal':
        y_dist = mv_normal(y_pred, pred_dim=y_true.shape[-1]) #(K, B, dim)
        loss = -y_dist.log_prob(y_true) #(K, B)
        # return log_prob.mean(0).sum() # sum will increase accuracy significantly, but KLD divergence will overfit
    
    elif loss_func=='mm_likelihood':
        loss = mm_likelihood(y_pred, y_true)

    elif loss_func=='normal':
        loss = -normal(y_pred).log_prob(y_true)

    else:
        loss = eval(loss_func)(y_true, y_pred)

    if K>1:
        loss, min_indices = torch.min(loss, dim=0) # min across samples

    # loss = loss.mean(0) * weight # (Batch, )
        
    if mode=='sum':
        return torch.sum(loss)
    elif mode=='average':
        return torch.mean(loss)
    else:
        return  loss 


def sample_traj(y_pred, loss_func, prob=False, sample_size=()):
    if loss_func == 'l2_loss':
        samples = y_pred
        log_prob = None

    elif loss_func == 'l1_loss':
        samples = y_pred
        log_prob = None
        
    elif loss_func=='margin_loss':
        samples = y_pred
        log_prob = None

    elif loss_func=='smoothl1_loss':
        samples = y_pred
        log_prob = None

    elif loss_func =='mv_normal':
        y_dist = mv_normal(y_pred)
        samples = y_dist.rsample()
        log_prob = y_dist.log_prob(samples)
    
    elif loss_func=='mm_likelihood':
        y_dist = bivariate_sample(y_pred)
        samples = y_dist.rsample()
        log_prob = y_dist.log_prob(samples)

    elif loss_func=='normal':
        y_dist = normal(y_pred)
        samples = y_dist.rsample(sample_size)
        log_prob = y_dist.log_prob(samples)

    elif loss_func=='bivariate' or loss_func=='mm_likelihood':
        y_dist = bivariate_sample(y_pred)
        samples = y_dist.rsample()
        log_prob = y_dist.log_prob(samples)

    else:
        raise Exception(f"Sorry, {loss_func} is not supported!")

    return samples, log_prob

def loss_func_param(loss_func, pred_dim=2):
    if loss_func=='mv_normal':
        return 3 * pred_dim
    elif loss_func=='bivariate':
        return 5
    elif loss_func=='normal':
        return 2*pred_dim
    elif loss_func=='mm_likelihood':
        return 5
    else:
        return pred_dim
        

class Loss(object):
    def __init__(self, name='', pred_states=['vel', 'acc'], pred_states_dim=[2, 2], **kwargs):
        self.name = name
        self.pred_states = pred_states
        self.pred_states_dim = pred_states_dim

    def compute(self):
        raise NotImplementedError("Subclasses should implement this!")

    def sample(self):
        raise NotImplementedError("Subclasses should implement this!")

class MVNormal(Loss):
    def __init__(self, pred_states=['vel', 'acc'], pred_states_dim=[2, 2], **kwargs):
        
        super().__init__('MVNormal', pred_states, pred_states_dim, **kwargs)

        self.loss_dims = {}
        for s, pred_dim in zip(pred_states, pred_states_dim):
            self.loss_dims[s] = 3 * pred_dim

    def compute(self, gt_state_dict, pred_state_dict, **kwargs):
        loss = 0
        for s in self.pred_states:
            loss = -mv_normal(pred_state_dict[s], self.pred_states_dim[s]).log_prob(gt_state_dict[s]).mean()
        return loss

    def sample(self, pred_state_dict):
        samples = {}
        log_probs = {}
        for s in self.pred_states:
            y_dist = mv_normal(pred_state_dict[s])
            samples[s] = y_dist.rsample()
            log_probs[s] = y_dist.log_prob(samples[s])

        return samples, log_probs


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg, device):
        super(MultiTaskLoss, self).__init__()

        self.cfg = cfg
        self.learn_loss_weights = cfg.learn_loss_weights
        if self.learn_loss_weights: 
            self.node_loss_wt = nn.ParameterDict({s:torch.nn.Parameter(torch.tensor(0.).to(device), requires_grad=True)  
                                                    for s in cfg.node_outputs
                                                    })
            if len(cfg.edge_outputs)>0:
                self.edge_loss_wt = nn.ParameterDict({s:torch.nn.Parameter(torch.tensor(0.).to(device), requires_grad=True)
                                                        for s in cfg.edge_outputs
                                                        })
        else:
            self.node_loss_wt = {s:torch.tensor(cfg.node_loss_wt.serialize[s], requires_grad=False) for s in cfg.node_outputs}
            if len(cfg.edge_outputs)>0:
                self.edge_loss_wt = {s:torch.tensor(cfg.edge_loss_wt.serialize[s], requires_grad=False) for s in cfg.edge_outputs}

    def forward(self, cfg, gt_graphs, logits_n, logits_e):
        node_loss, edge_loss = {}, {}
        
        # ground truth
        gt_ndata = {s:gt_graphs.ndata[s].unsqueeze(0) for s in cfg.node_outputs}

        # gt_ndata['pos'][..., 0] = gt_ndata['pos'][..., 0]/data_stats[cfg.dset]['std'][0]
        # gt_ndata['pos'][..., 1] = gt_ndata['pos'][..., 1]/data_stats[cfg.dset]['std'][1]
        
        node_weights = gt_graphs.ndata['ftl'].unsqueeze(1)/cfg.pred_seq_len

        # node loss

        # mask lane nodes
        if cfg.include_lane:
            lane_mask = gt_graphs.ndata['cid']!=NODE_TYPES.index('LANE')
            if lane_mask.sum()>0:
                gt_y = gt_y[lane_mask]
                logits_y = logits_y[:, lane_mask, :]

        for s in cfg.node_outputs:
            nloss = compute_loss(gt_ndata[s], logits_n[s], cfg.node_loss.serialize[s], weight=node_weights, mode=cfg.loss_mode)
            if self.learn_loss_weights:
                precision = torch.exp(-self.node_loss_wt[s])
                node_loss[s] = nloss*precision + self.node_loss_wt[s]
            else:
                node_loss[s] = nloss * self.node_loss_wt[s]

        if len(cfg.edge_outputs)>0:
            gt_edata = {s:gt_graphs.edata[s].unsqueeze(0) for s in cfg.edge_outputs}

            for s in cfg.edge_outputs:
                eloss = compute_loss(gt_edata[s], logits_e[s], cfg.edge_loss.serialize[s], weight=1., mode=cfg.loss_mode)
                if self.learn_loss_weights:
                    precision = torch.exp(-self.edge_loss_wt[s])
                    edge_loss[s] = eloss*precision + self.edge_loss_wt[s]
                else:
                    edge_loss[s] = eloss*self.edge_loss_wt[s]

        return node_loss, edge_loss
