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

        epoch_ade = 0
        epoch_fde = 0
        epoch_ade_best = 0
        epoch_fde_best = 0
        epoch_ade_mean = 0
        epoch_fde_mean = 0
        # epoch_edge_diff = 0

        for iter, (seq_graphs, obsv_graphs, trgt_graphs) in enumerate(dataloader):
            obsv_graphs = obsv_graphs.to(device)
            trgt_graphs = trgt_graphs.to(device)

            # remove nodes from trgt_graphs that are not present in obsv graphs
            gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
            # make sure there is at least one edges
            if not gt_graphs.number_of_edges()>1:
                continue

            if not obsv_graphs.number_of_edges()>1:
                continue

            # common traj in obsv and gt
            comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))
            # lane_tids = [tid for tid in comm_traj if gt_graphs.ndata['cid'][gt_graphs.ndata['tid']==tid].unique()==NODE_TYPES.index('LANE')]
            lane_tids = gt_graphs.ndata['tid'][gt_graphs.ndata['cid']==NODE_TYPES.index('LANE')].unique()
            
            # prepare
            gx, xx, ex = prepare_inputs(obsv_graphs, cfg.node_inputs, cfg.edge_inputs)

            # predict
            logits_y, _ = model(gx, xx, ex, comm_traj=comm_traj, num_samples=cfg.K_eval, device=device)

            # sample 
            pred_states = {s:sample_traj(logits_y[s], cfg.node_loss.serialize[s]) for s in cfg.node_outputs}

            # last pos
            last_pos = torch.stack([obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid][-1] for tid in comm_traj], dim=0) #[num_traj, 2]
            
            # pred pos
            pred_pos = leapfrogIntegrator(last_pos, pred_states, cfg.dt)

            # break
            gt_pos, gt_masks = graph_data_to_traj(gt_graphs, gt_graphs.ndata['pos'], seq_len=cfg.pred_seq_len, traj_ids=comm_traj)

            log_probs = pred_states['vel'][1]
            if log_probs is not None:
                log_probs = F.softmax(torch.sum(log_probs * (1-gt_masks.repeat(log_probs.shape[0], 1, 1)), dim=-1)/torch.sum(1-gt_masks, dim=1), dim=0)

            ade, fde = compute_ade_fde_k(gt_pos, pred_pos, log_probs, gt_masks, top_k=cfg.K_eval) 
            
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

def train_val_epoch(cfg, model, optimizer, dataloader, device, epoch, phase, history=None, writer=None, **kwargs):

    multi_tasks_loss = kwargs['multi_tasks_loss']

    if phase=='train':
        model.train()
        multi_tasks_loss.train()
    else:
        model.eval()
        multi_tasks_loss.eval()

    hist = defaultdict(list) if not history else history  
    epoch_loss = 0
    epoch_kld_loss = 0
    epoch_node_loss = 0
    epoch_edge_loss = 0
    epoch_ade_min = 0
    epoch_goal_loss = 0

    epoch_node_loss_dict = {s:0.0 for s in cfg.node_outputs}
    epoch_edge_loss_dict = {s:0.0 for s in cfg.edge_outputs}

    pbar = tqdm(total=len(dataloader), position=0)
    for iter, (seq_graphs, obsv_graphs, trgt_graphs) in enumerate(dataloader):
        pbar.update(1)

        seq_graphs = seq_graphs.to(device)
        obsv_graphs = obsv_graphs.to(device)
        trgt_graphs = trgt_graphs.to(device)
        
        # remove nodes from trgt_graphs that are not present in obsv graphs
        gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
        if not seq_graphs.number_of_edges()>1:
            continue

        if not obsv_graphs.number_of_edges()>1:
            continue

        if not gt_graphs.number_of_edges()>1:
            continue

        # common traj in obsv and gt
        comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))

        # prepare inputs/standardize   
        gx, xx, ex = prepare_inputs(obsv_graphs, cfg.node_inputs, cfg.edge_inputs)
        gy, yy, ey = prepare_inputs(seq_graphs, cfg.node_inputs, cfg.edge_inputs)

        #predict
        logits_y, KLD = model(gx, xx, ex, gy, yy, ey, gt=gt_graphs, comm_traj=comm_traj, num_samples=cfg.K, device=device)

        # convert to graph data
        _, node_idx, time_idx = traj_to_graph_data(gt_graphs, traj_ids=comm_traj)
        logits_n = {s:logits_y[s][..., node_idx, time_idx, :] for s in cfg.node_outputs}# (K, num_nodes, 6)

        node_loss, edge_loss = multi_tasks_loss(cfg, gt_graphs, logits_n, logits_e=None)

        # # min ade k
        # min_ade, min_indices = min_ade_nodes(gt_graphs, logits_y) # logits_y[..., :2] is used to compute min_err
        # epoch_ade_min += min_ade.detach().item()

        loss = sum([node_loss[s] for s in cfg.node_outputs])

        if phase=='train':
            loss += KLD*cfg.kld_loss_wt 

            optimize(cfg, loss, optimizer)
            
            # detach
            loss = loss.detach()

            KLD = KLD.detach().item()
            node_loss = {s:l.detach() for s, l in node_loss.items()}


        # add epoch loss
        epoch_loss += loss
        epoch_kld_loss += KLD
        epoch_node_loss += sum([l  for _, l in node_loss.items()])
        epoch_edge_loss += sum([l  for _, l in edge_loss.items()])

        for s in cfg.node_outputs:
            epoch_node_loss_dict[s] += node_loss[s]

        for s in cfg.edge_outputs:
            epoch_edge_loss_dict[s] += edge_loss[s]
            
        # print(f"{node_loss}, {KLD}, {loss}")

    pbar.close()

    hist[f'{phase}_loss'].append(epoch_loss.item()/(iter+1))
    hist[f'{phase}_kld_loss'].append(epoch_kld_loss/(iter+1))
    hist[f'{phase}_node_loss'].append(epoch_node_loss.item()/(iter+1))

    for s in cfg.node_outputs:
        hist[f'{phase}_{s}_loss'].append(epoch_node_loss_dict[s].item()/(iter+1))

    if len(cfg.edge_outputs)>0:
        hist[f'{phase}_edge_loss'].append(epoch_edge_loss.item()/(iter+1))

        for s in cfg.edge_outputs:
            hist[f'{phase}_{s}_loss'].append(epoch_edge_loss_dict[s].item()/(iter+1))

    return hist
    

