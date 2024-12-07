#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:39:55 2021

@author: loc
"""
import math
import torch


import torch.nn as nn
import torch.distributions as tdist
torch.distributions.Distribution.set_default_validate_args(True)

import torch.nn.functional as F

# Reference from cvae_dsprites that uses Student-T
# https://github.com/iffsid/disentangling-disentanglement/blob/public/src/models/vae_dsprites.py
_EPSILON = 1e-9

class VAE(nn.Module):
    def __init__(self, net_params, **kwargs):
        super().__init__()
        self.name = 'VAE'
        
        self.net_params = net_params

        self.pz_x = getattr(tdist, 'StudentT')
        self.pa_ex = getattr(tdist, 'StudentT')
        
        self.qz_x_y = getattr(tdist, 'Normal')
        self.qa_ex_ey = getattr(tdist, 'Normal')
        
        self.py_z_x = getattr(tdist, 'Normal')

    def p_dist_from_h(self, h, latent_dim, p_dist, scale=0.03):
        # model p(z|x)
        mean = h[..., :latent_dim]
        logvar =  h[..., latent_dim:]

        # mean = mean - mean.mean(dim=-1, keepdim=True)
        logvar = logvar.mul(0.5).exp() + _EPSILON

        assert torch.all(logvar>0), f'logvar is not positives: {logvar}'

        if p_dist.__name__ == 'StudentT':
            logvar = torch.sqrt(scale * logvar.size(-1) * F.softmax(logvar, dim=1)) + _EPSILON
            try:
                return p_dist(25.0, mean, logvar) # df>10 is almost like normal distribution
            except ValueError as e:
                print(e, mean, logvar)
                raise
        else:
            return p_dist(mean, logvar)

    def q_dist_from_h(self, h, latent_dim, q_dist):
        # model q(z|x, y)
        mean = h[..., :latent_dim]
        # mean = mean - mean.mean(dim=-1, keepdim=True)
        
        logvar = h[..., latent_dim:]
        logvar = logvar.mul(0.5).exp() + _EPSILON
        # logvar = F.softmax(logvar, dim=1)

        return q_dist(mean, logvar)


    def kl_divergence(self, p, q, samples=None):
        B, D = q.batch_shape

        if (type(p), type(q)) in torch.distributions.kl._KL_REGISTRY:
            return tdist.kl_divergence(p, q).sum(-1).sum() # (num_batch, dim)
        else:
            if samples is None:
                K = 12
                samples = p.rsample(torch.Size([K])) if p.has_rsample else p.sample(torch.Size([K])) 

            lpz = p.log_prob(samples).sum(-1)  #(K, num_batch,)
            lqz = q.log_prob(samples).sum(-1)
            
            kld = (lpz - lqz) # (num_batch, dim), mean over samples

            return kld.mean(0).sum()

    def inc_kl_divergence(self, pz, qz_x, samples=None):
        B, D = qz_x.loc.shape

        _zs = pz.rsample(torch.Size([B])) #(B, B, D)

        lpz = pz.log_prob(_zs).sum(-1).squeeze(-1) #(B, B)

        lqz = log_mean_exp(qz_x.log_prob(_zs).sum(-1), dim=1).unsqueeze(1) #(B, )

        inc_kld = lpz - lqz #(B, B)

        return inc_kld.mean(0).sum()

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))
