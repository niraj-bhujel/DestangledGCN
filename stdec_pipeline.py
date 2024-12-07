#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:29:48 2020

@author: loc
"""

import os
import sys
import copy

import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.states import * 
from data.data_stats import data_stats
from data.preprocess_utils import *
from model.losses import *

from utils.graph_utils import *
from utils.train_utils import *
from utils.misc import *
from utils.metrics import *
from utils.config import *

def predict_stdec_inputs(cfg, obsv_graphs, gt_graphs, comm_traj, prediction_model, device, phase='train',):
    if prediction_model is not None:
        prediction_model.eval()

    #%%

    # if phase=='train':
    #     # goals = gt_graphs.ndata['goal'].unsqueeze(0)
    #     # goals = goals + 0.5*torch.rand_like(goals)
    #     _, ndata, edata = prepare_inputs(gt_graphs, cfg.node_inputs, cfg.edge_inputs)
        
    #     ndata['pos'] = gt_graphs.ndata['pos']
    #     ndata['hed'] = gt_graphs.ndata['hed']
    #     ndata['speed'] = gt_graphs.ndata['speed']

    #     ndata = {s:v.unsqueeze(0) for s, v in ndata.items()}
    #     edata = {s:v.unsqueeze(0) for s, v in edata.items()}
    # else:

    init_pos = torch.stack([pos[-1] for pos in get_ndata(obsv_graphs, 'pos', comm_traj)]) #[num_traj, 2]
    init_vel = torch.stack([vel[-1] for vel in get_ndata(obsv_graphs, 'vel', comm_traj)])
    final_pos = torch.stack([pos[-1] for pos in get_ndata(gt_graphs, 'pos', comm_traj)])
    
    # convert to graph data
    _, node_idx, time_idx = traj_to_graph_data(gt_graphs, traj_ids=comm_traj)
    init_pos = init_pos.unsqueeze(1).unsqueeze(0).repeat(1, 1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]
    init_vel = init_vel.unsqueeze(1).unsqueeze(0).repeat(1, 1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]
    final_pos = final_pos.unsqueeze(1).unsqueeze(0).repeat(1, 1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]
    
    pred_states = {}

    if cfg.trajdec=='ConstVel':
        pred_states['vel'] = init_vel # constant velocity 
        
    else:
        # prepare
        gx, xx, ex = prepare_inputs(obsv_graphs, prediction_model.node_inputs, prediction_model.edge_inputs)

        # predict
        logits_y, _ = prediction_model(gx, xx, ex, comm_traj=comm_traj, num_samples=cfg.K, device=device)

        # sample 
        pred_states['vel'] = sample_traj(logits_y['vel'], 'mv_normal')[0][..., node_idx, time_idx, :]

    # pred pos
    pred_states['pos'] = init_pos + pred_states['vel']*gt_graphs.ndata['ntx'].unsqueeze(0).sub(cfg.obsv_seq_len-1)*cfg.dt

    # get best pred if multiple samples
    if pred_states['pos'].shape[0]>1 and phase!='eval':
        min_ade, min_indices = min_ade_nodes(gt_graphs.ndata['pos'] , pred_states['pos'])
        pred_states = {k:v.gather(0, min_indices[None, :, None].repeat(1, 1, 2)) for k, v in pred_states.items()}
    
    num_samples = pred_states['vel'].shape[0]

    # cv goal pos
    final_pos = torch.stack([pred_states['pos'].squeeze(0)[gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
    # final_pos = torch.stack([gt_graphs.ndata['pos'][gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
    goals = final_pos.unsqueeze(0).unsqueeze(2).repeat(1, 1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]

    if cfg.sample_goal:
        # goals = torch.stack([pred_states['pos'].squeeze(0)[gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
        goals = torch.distributions.normal.Normal(final_pos, scale=1).sample((cfg.K_eval,)) # goals must be (num_traj, 2)
        goals = goals.unsqueeze(2).repeat(1, 1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]
        
        # recompute goal vel
        pred_states['vel'] = (goals - init_pos)/(gt_graphs.ndata['ftl'].view(1, -1, 1)*cfg.dt)
        pred_states['pos'] = init_pos + pred_states['vel']*gt_graphs.ndata['ntx'].sub(cfg.obsv_seq_len-1).view(1, -1, 1)*cfg.dt

        # get best pred if multiple samples
        if phase!='eval':
            final_ntx = gt_graphs.ndata['ntx'].flatten()==gt_graphs.ndata['ftl'].add(cfg.obsv_seq_len-1)
            _, min_indices = min_fde_nodes(gt_graphs.ndata['pos'], pred_states['pos'], final_ntx)
            # _, min_indices = min_ade_nodes(gt_graphs.ndata['pos'], pred_states['pos'])
            min_indices_rpt =  min_indices[None, :, None].repeat(1, 1, 2)
            pred_states = {k:v.gather(0, min_indices_rpt) for k, v in pred_states.items()}
            goals = goals.gather(0, min_indices_rpt)

    goal_dir = goals - init_pos
    goal_norm = torch.linalg.norm(goal_dir, dim=-1, keepdim=True)
    # goal_hed = goal_dir
    pred_states['hed'] = torch.where(goal_norm>0, goal_dir/goal_norm, goal_dir)
    # pred_states['hed'] = init_vel/(torch.linalg.norm(init_vel, dim=-1, keepdim=True) + 1e-9)
    pred_states['vnorm'] = pred_states['vel'].norm(dim=-1, keepdim=True)
    pred_states['dir'] = torch.where(pred_states['vnorm']>0, pred_states['vel']/pred_states['vnorm'], pred_states['vel'])
    pred_states['ntx'] = gt_graphs.ndata['ntx'].unsqueeze(0).repeat(num_samples, 1, 1)
    # pred_states['hed'] = gt_graphs.ndata['hed'].unsqueeze(0).repeat(num_samples, 1, 1)

    # prepare node inputs
    ndata = {}
    for s in cfg.node_inputs:
        if s in NODE_INFO:
            ndata[s] = gt_graphs.ndata[s].view(1, -1, 1).repeat(num_samples, 1, 1)
        else:
            ndata[s] = pred_states[s]

    # prepare edge inputs
    edata = edata_from_ndata(gt_graphs, pred_states['pos']) # note that standardized ndata also standardize edata
    edata = {s:edata[s] for s in cfg.edge_inputs}

    if cfg.net.layer=='GatedGCNLayerSF' or cfg.net.layer=='GatedGCNLSPELayer':
        ndata['speed'] = torch.linalg.norm(init_vel, dim=-1, keepdims=True)
        ndata['hed'] = pred_states['hed']

        ndata['pos'] = pred_states['pos']
        ndata['pos'][..., 0] = ndata['pos'][..., 0]/data_stats[cfg.dset]['std'][0]
        ndata['pos'][..., 1] = ndata['pos'][..., 1]/data_stats[cfg.dset]['std'][1]
    
    # ndata = {s:standardize_nodes(gt_graphs, ndata[s], cfg.input_nodes_std_param[s]) for s in cfg.node_inputs}
    
    # quantize
    ndata = {s:quantize_tensor(v, QLEVELS[s]) for s, v in ndata.items()}
    edata = {s:quantize_tensor(v, QLEVELS[s]) for s, v in edata.items()}

    return gt_graphs, ndata, edata, num_samples
    
            
def evaluate_model(cfg, model, dataloader, device, history=None, return_raw_data=False, **kwargs):
    
    hist = defaultdict(list) if not history else history  
    raw_data_dict = defaultdict(dict)
    
    model.eval()
    prediction_model = kwargs['prediction_model']
    
    with torch.no_grad():

        epoch_ade = 0
        epoch_fde = 0
        epoch_ade_best = 0
        epoch_fde_best = 0
        epoch_ade_mean = 0
        epoch_fde_mean = 0
        
        for iter, (_, obsv_graphs, trgt_graphs) in enumerate(dataloader):
            obsv_graphs = obsv_graphs.to(device)
            trgt_graphs = trgt_graphs.to(device)

            # remove nodes from trgt_graphs that are not present in obsv graphs
            gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)

            # make sure there is at least one edges
            if not gt_graphs.number_of_edges()>1:
                continue

            # if cfg.dataset=='waymo':
            #     gt_graphs.ndata['pos'] = gt_graphs.ndata['pos'] - gt_graphs.ndata['pos'].mean(-1, keepdims=True)

            # common traj in obsv and gt
            comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))

            lane_tids = gt_graphs.ndata['tid'][gt_graphs.ndata['cid']==NODE_TYPES.index('LANE')].unique()
            
            # if only lane nodes, skip
            if len(comm_traj)==len(lane_tids):
                continue

            # prepare stdec inputs
            gt_graphs, ndata, edata, num_samples = predict_stdec_inputs(cfg, obsv_graphs, gt_graphs, comm_traj, prediction_model, device, phase='eval')
            
            # predict
            logits_n, logits_e = model(gt_graphs, ndata, edata, comm_traj=comm_traj, num_samples=num_samples, device=device)
            # logits_n, logits_e = stdec_faster(model, gt_graphs, ndata, edata, device=device)
            # logits_n = logits_to_dict(logits_n, state_dims=cfg.output_nodes_dim)

            pred_states = {}
            for s in cfg.node_outputs:
                pred = sample_traj(logits_n[s], cfg.node_loss.serialize[s])[0]
                pred = graph_data_to_traj(gt_graphs, pred, seq_len=cfg.pred_seq_len, traj_ids=comm_traj)
                pred_states[s] = pred

            init_pos = torch.stack([obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid][-1] for tid in comm_traj]) #[num_traj, 2]
            
            # pred pos
            pred_pos = leapfrogIntegrator(init_pos, pred_states, cfg.dt)
            
            # gt pos
            gt_pos, gt_masks = graph_data_to_traj(gt_graphs, gt_graphs.ndata['pos'], seq_len=cfg.pred_seq_len, traj_ids=comm_traj)

            probs = torch.ones(pred_pos.shape[0], 1)

            ade, fde = compute_ade_fde_k(gt_pos, pred_pos, probs, gt_masks)  # ranked ade/fde
            
            if cfg.include_lane:
                vehicle_node_idx = torch.tensor([i for i,traj in enumerate(comm_traj) if traj not in lane_tids]).to(device)
                # vehicle_nodes = [traj not in lane_tids for traj in comm_traj] # 1 indicate vehicle nodes 0 indicate lane
                ade = ade.gather(1, vehicle_node_idx.repeat(ade.shape[0], 1))
                fde = fde.gather(1, vehicle_node_idx.repeat(fde.shape[0], 1))

            epoch_ade += torch.mean(ade[0]).item()
            epoch_fde += torch.mean(fde[0]).item()
            epoch_ade_best += torch.mean(ade.min(0).values).item()
            epoch_fde_best += torch.mean(fde.min(0).values).item()

            epoch_ade_mean += torch.mean(ade).item()
            epoch_fde_mean += torch.mean(fde).item()

            if return_raw_data:
                raw_data_dict[iter]['pred_pos'] = pred_pos
                raw_data_dict[iter]['pred_states'] = pred_states
                raw_data_dict[iter]['log_probs'] = log_probs
                raw_data_dict[iter]['comm_traj'] = comm_traj
                raw_data_dict[iter]['obsv_graphs'] = obsv_graphs
                raw_data_dict[iter]['gt_graphs'] = gt_graphs
                raw_data_dict[iter]['ade'] = ade
                raw_data_dict[iter]['fde'] = fde
        
        hist['test_ade'].append(epoch_ade/(iter+1))
        hist['test_fde'].append(epoch_fde/(iter+1))
        hist['test_ade_best'].append(epoch_ade_best/(iter+1))
        hist['test_fde_best'].append(epoch_fde_best/(iter+1))
        hist['test_ade_mean'].append(epoch_ade_mean/(iter+1))
        hist['test_fde_mean'].append(epoch_fde_mean/(iter+1))

    return hist, raw_data_dict

def train_val_epoch(cfg, model, optimizer, dataloader, device, epoch, phase, history=None, **kwargs):

    prediction_model = kwargs['prediction_model']
    multi_tasks_loss = kwargs['multi_tasks_loss']

    if phase=='train':
        model.train()
        multi_tasks_loss.train()
    else:
        model.eval()
        multi_tasks_loss.eval()
        
    hist = defaultdict(list) if not history else history  

    epoch_loss = 0
    epoch_node_loss = 0
    epoch_edge_loss = 0

    epoch_node_loss_dict = {s:0.0 for s in cfg.node_outputs}
    epoch_edge_loss_dict = {s:0.0 for s in cfg.edge_outputs}


    pbar = tqdm(total=len(dataloader), desc='%6s'%phase, position=0, leave=True)
    for iter, (_, obsv_graphs, trgt_graphs) in enumerate(dataloader):
        pbar.update(1)
        obsv_graphs = obsv_graphs.to(device)
        trgt_graphs = trgt_graphs.to(device)

        # remove nodes from trgt_graphs that are not present in obsv graphs
        gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
        if not gt_graphs.number_of_edges()>1:
            continue

        # if cfg.dataset=='waymo':
        #     gt_graphs.ndata['pos'] = gt_graphs.ndata['pos'] - gt_graphs.ndata['pos'].mean(-1, keepdims=True)

        # common traj in obsv and gt
        comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))
        
        # prepare stdec inputs
        with torch.no_grad():
            gt_graphs, ndata, edata, num_samples = predict_stdec_inputs(cfg, obsv_graphs, gt_graphs, comm_traj, prediction_model, device, phase='train')
                
        # predict
        logits_n, logits_e = model(gt_graphs, ndata, edata, comm_traj=comm_traj, num_samples=num_samples, device=device) 

        # node_loss, edge_loss = loss_step(cfg, gt_graphs, logits_n, logits_e)
        node_loss, edge_loss = multi_tasks_loss(cfg, gt_graphs, logits_n, logits_e)

        loss = sum([node_loss[s] for s in cfg.node_outputs])
        if len(cfg.edge_outputs)>0:
            loss += sum([edge_loss[s] for s in cfg.edge_outputs])

        if phase=='train':
            optimize(cfg, loss, optimizer)
            
            # detach
            loss = loss.detach()
            node_loss = {s:l.detach() for s, l in node_loss.items()}
            edge_loss = {s:l.detach() for s, l in edge_loss.items()}

        # add epoch loss
        epoch_loss += loss
        epoch_node_loss += sum([l  for _, l in node_loss.items()])
        epoch_edge_loss += sum([l  for _, l in edge_loss.items()])

        for s in cfg.node_outputs:
            epoch_node_loss_dict[s] += node_loss[s]

        for s in cfg.edge_outputs:
            epoch_edge_loss_dict[s] += edge_loss[s]

    pbar.close()


    hist[f'{phase}_loss'].append(epoch_loss.item()/(iter+1))
    hist[f'{phase}_node_loss'].append(epoch_node_loss.item()/(iter+1))
    hist[f'{phase}_edge_loss'].append(epoch_edge_loss.item()/(iter+1))

    for s in cfg.node_outputs:
        hist[f'{phase}_{s}_loss'].append(epoch_node_loss_dict[s].item()/(iter+1))

    for s in cfg.edge_outputs:
        hist[f'{phase}_{s}_loss'].append(epoch_edge_loss_dict[s].item()/(iter+1))


    return hist