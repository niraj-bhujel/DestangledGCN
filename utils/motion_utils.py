#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:57:17 2021

@author: loc
"""
import math
import torch
import numpy as np

def vel_clamping(vel, max_vel=15, max_speed=20, method='max', obsv_vel=None, norm=False):
    '''
    Clamped velocity to max_vel, if method = max 
    If not method==max, compute max vel for each individual peds from obsv_vel and perform clamp
    This function can slow be used for clamping other states like acc.
    vel: array of shape [K, N, 12, 2]
    norm: if norm, clip using norm. NOTE: This will clip in both x, y direction with same norm, which is sometimes undesirable. 
    '''
    if method=='max':

        if norm:
            speed = torch.norm(vel, dim=-1, keepdim=True)     
            new_vel = torch.where(speed>max_speed, vel/torch.abs(vel)*max_speed, vel)
            
        else:
            new_vel = vel.clamp(-max_vel, max_vel)
    
    # this one gives the best accuracy
    elif method=='individual_max':
    
        if norm:
            max_speed = torch.stack([v.norm(dim=-1, keepdim=True).max() for v in obsv_vel], dim=0)[None, :, None, None]
            speed = torch.vnorm(vel, dim=-1, keepdim=True)
            
            new_vel = torch.where(speed>max_speed, vel/torch.abs(vel)*max_speed, vel)
            
        else:
            # absolute max
            max_vel = [v.abs().max(0).values for v in obsv_vel]
            max_vel = torch.stack([v for v in max_vel], dim=0)[None, :, None, :]
            
            new_vel = torch.where(vel.abs()>max_vel, max_vel*vel.sign(), vel)

    elif method=='average':
        # absolute average
        avg_vel = torch.stack([v.abs().mean(0) for v in obsv_vel], dim=0)[None, :, None, :]
        new_vel = torch.where(vel.abs()>avg_vel, avg_vel*vel.sign(), vel)
        
    elif method=='exponential_average':
        weights = [torch.exp(torch.linspace(-1., 0., len(v))) for v in obsv_vel]
        weights = [w.unsqueeze(1).to(vel.device) for w in weights]
        # weights = [w/w.sum() for w in weights]

        exp_avg_vel = [torch.sum(v.abs() * w, dim=0)/w.sum(0) for v, w in zip(obsv_vel, weights)]
        exp_avg_vel = torch.stack(exp_avg_vel, dim=0)[None, :, None, :]
        
        new_vel = torch.where(vel.abs()>exp_avg_vel, exp_avg_vel*vel.sign(), vel)
        
    elif method=='last':
        last_vel = torch.stack([v[-1].abs() for v in obsv_vel], dim=0)[None, :, None, :]
        new_vel = torch.where(vel.abs()>last_vel, last_vel*vel.sign(), vel)
        
    else:
        raise ValueError('method %s not one of the valid methods (max, individual_max, average, exponential_average, last)'%method)
        
    return new_vel

def acc_clamping(vel, max_acc=1.5, method='max', obsv_acc=None, dt=0.5):
    '''
    Dampened velocity based on accleration, First convert velocity to accleration. 
    vel: array of shape [K, N, 12, 2]
    obsv_acc: list of tensor, each with shape (T, 2)
    '''
    acc = 1/dt * (vel[:, :, 1:, :] - vel[:, :, :-1, :])
    acc = torch.cat([torch.zeros_like(acc)[:, :, 0:1, :], acc], dim=2)

    # clamped vel can be used for accleration as well
    new_acc = vel_clamping(acc, max_acc, method=method, obsv_vel=obsv_acc)
    
    new_vel = vel[:, :, 0:1, :]  + dt*new_acc.cumsum(2)
    
    return new_vel
