#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:29:48 2020

@author: loc
"""
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.states import * 

from model.losses import *

from utils.misc import *
from utils.metrics import *
from utils.graph_utils import *
from utils.train_utils import *
from utils.config import *

def evaluate_model(cfg, model, dataloader, device, history=None, return_raw_data=False, writer=None, **kwargs):
        
    hist = defaultdict(list) if not history else history  
    raw_data_dict = defaultdict(dict)
    
    model.eval()
    
    with torch.no_grad():

        epoch_goal_err = 0

        epoch_ade = 0
        epoch_fde = 0
        epoch_ade_best = 0
        epoch_fde_best = 0

        for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):
            obsv_graphs = obsv_graphs.to(device)
            trgt_graphs = trgt_graphs.to(device)

            # remove nodes from trgt_graphs that are not present in obsv graphs
            gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
            # make sure there is at least one edges
            if not gt_graphs.number_of_edges()>1:
                continue

            if not obsv_graphs.number_of_edges()>1:
                continue

            if not trgt_graphs.number_of_edges()>1:
                continue

            # common traj in obsv and gt
            comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))
            # lane_tids = [tid for tid in comm_traj if gt_graphs.ndata['cid'][gt_graphs.ndata['tid']==tid].unique()==NODE_TYPES.index('LANE')]

            # prepare inputs/standardize
            gx = obsv_graphs
            xx = []
            for tid in comm_traj:
                traj_states = torch.cat([gx.ndata[s][gx.ndata['tid']==tid][-1] for s in cfg.goal_inputs], dim=-1)
                xx.append(traj_states)
            
            xx = torch.stack(xx)
                

            #predict
            logits_y = model(gx, xx, comm_traj=comm_traj, num_samples=cfg.K_eval)
            
            pred_goal, log_probs = sample_traj(logits_y, cfg.goal_loss)

            last_pos = torch.stack([obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
            final_pos = torch.stack([gt_graphs.ndata['pos'][gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
            # gt_goal = final_pos - last_pos

            pred_pos = last_pos + pred_goal
            goal_err = torch.linalg.norm(final_pos - pred_pos, dim=-1).mean()

            epoch_goal_err += goal_err.item()

        hist['test_ade'].append(epoch_ade/(iter+1))
        hist['test_fde'].append(epoch_fde/(iter+1))
        hist['test_ade_best'].append(epoch_ade_best/(iter+1))
        hist['test_fde_best'].append(epoch_fde_best/(iter+1))
        hist['test_goal_err'].append(epoch_goal_err/(iter+1))

    return hist, raw_data_dict
    
def val_epoch(cfg, model, dataloader, device, epoch, history=None, writer=None, **kwargs):
    
    model.eval()
    
    hist = defaultdict(list) if not history else history

    epoch_goal_loss = 0
    epoch_loss = 0
    with torch.no_grad():
        for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):
            obsv_graphs = obsv_graphs.to(device)
            trgt_graphs = trgt_graphs.to(device)
            
            # remove nodes from trgt_graphs that are not present in obsv graphs
            gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
            # make sure there is at least one edges
            if not gt_graphs.number_of_edges()>1:
                continue

            if not obsv_graphs.number_of_edges()>1:
                continue

            if not trgt_graphs.number_of_edges()>1:
                continue

            # common traj in obsv and gt
            comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))        
            # prepare            

            # prepare inputs/standardize
            gx = obsv_graphs
            xx = []
            for tid in comm_traj:
                traj_states = torch.cat([gx.ndata[s][gx.ndata['tid']==tid][-1] for s in cfg.goal_inputs], dim=-1)
                xx.append(traj_states)
            
            xx = torch.stack(xx)
                

            #predict
            logits_y = model(gx, xx, comm_traj=comm_traj, num_samples=cfg.K)
            

            last_pos = torch.stack([obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
            final_pos = torch.stack([gt_graphs.ndata['pos'][gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
            goal_loss = compute_loss(final_pos-last_pos, logits_y, loss_func=cfg.goal_loss)
            epoch_goal_loss += goal_loss.item()


    hist['val_goal_loss'].append(epoch_goal_loss/(iter+1)) # required by lr scheduler
    hist['val_loss'].append(epoch_goal_loss/(iter+1))

    return hist

def train_epoch(cfg, model, optimizer, dataloader, device, epoch, history=None, writer=None, **kwargs):
    
    model.train()
        
    hist = defaultdict(list) if not history else history  
    
    epoch_goal_loss = 0
    pbar = tqdm(total=len(dataloader), position=0)
    for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):
        pbar.update(1)

        obsv_graphs = obsv_graphs.to(device)
        trgt_graphs = trgt_graphs.to(device)
        
        # remove nodes from trgt_graphs that are not present in obsv graphs
        gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
        if not gt_graphs.number_of_edges()>1:
            continue

        if not obsv_graphs.number_of_edges()>1:
            continue

        if not trgt_graphs.number_of_edges()>1:
            continue

        # common traj in obsv and gt
        comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))


        # prepare inputs/standardize
        gx = obsv_graphs
        xx = []
        for tid in comm_traj:
            traj_states = torch.cat([gx.ndata[s][gx.ndata['tid']==tid][-1] for s in cfg.goal_inputs], dim=-1)
            xx.append(traj_states)
        
        xx = torch.stack(xx)
            

        #predict
        logits_y = model(gx, xx, comm_traj=comm_traj, num_samples=cfg.K)
        

        last_pos = torch.stack([obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
        final_pos = torch.stack([gt_graphs.ndata['pos'][gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
        goal_loss = compute_loss(final_pos-last_pos, logits_y, loss_func=cfg.goal_loss)
        epoch_goal_loss += goal_loss.detach().item()

        goal_loss.backward()
        optimizer.step()    
        optimizer.zero_grad()
        
    pbar.close()

    hist['train_goal_loss'].append(epoch_goal_loss/(iter+1))

    return hist
    

