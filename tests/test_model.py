#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:20:35 2020

@author: dl-asoro
"""

import warnings
# warnings.filterwarnings("ignore")

import time
import yaml
import math
import numpy as np
from tqdm import tqdm
import itertools
import sys
import os
import dgl.function as fn
import matplotlib
matplotlib.use('Qt5Agg')
import importlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.autograd.profiler as profiler

from torch import autograd 
from torch.utils.data import DataLoader


# os.chdir(os.path.dirname(os.path.realpath(__file__)))
ROOT_DIR1 = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath('../')
print('ROOT_DIR', ROOT_DIR)
# os.environ['ROOT_DIR'] = ROOT_DIR
src_dir = ROOT_DIR + '/src'
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    

from argument_parser import parse_args

from data.states import *
from utils.config import Config
from model.losses import loss_func_param

args = parse_args()

# args.config = './configs/pedestrian_gcn.py'
# args.config = './configs/pedestrian_dgcn.py'
# args.config = './configs/pedestrian_stdec.py'
args.config = './configs/pedestrian_stdec_sf.py'
# args.config = './configs/waymo/gcn_config.py'
# args.config = './configs/waymo/sa_gcn_config.py'
# args.config = './configs/waymo/san_config.py'

cfg = Config.fromfile(args.config)
    
cfg_dict = cfg.serialize
print(cfg_dict)

torch.cuda.init()
# device = torch.device('cuda:0')
device = torch.device('cpu')
# args.update(cfg_dict)

#setup seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
    
#%%
from data.dataset import PedestriansDataset, NuScenesDataset, WaymoDataset

all_dsets = {'pedestrians':PedestriansDataset, 'nuscenes': NuScenesDataset, 'waymo': WaymoDataset}
cfg.node_types = ['PEDESTRIAN']
cfg.include_lane = False
version = ''
# version = 'mini'
# version = 'full'
# data_dir = '../../datasets/nuscenes/processed_data'
data_dir = '../datasets/pedestrians/eth/'
phase = 'train'
shuffle = True if phase=='train' else False
dataset = all_dsets[cfg.dataset](data_dir,
                                phase,
                                interaction_radius=INTERACTION_RADIUS,
                                version=version,
                                # node_type=cfg.node_type,
                                node_types=cfg.node_types, # for trial < 9
                                obsv_len=cfg.obsv_seq_len,
                                pred_len=cfg.pred_seq_len, 
                                min_obsv_len=cfg.min_obsv_len,
                                min_seq_len=cfg.min_seq_len,
                                dt=cfg.dt,
                                skip=cfg.skip if phase=='train' else 0,
                                aug_prob=cfg.augment_prob if phase=='train' else 0,
                                min_agents = cfg.min_agents,
                                include_lane=cfg.include_lane,
                                # pos_enc=cfg.pos_enc,
                                # pos_enc_dim=cfg.pos_enc_dim,
                                fov_graph=False,
                                preprocess=False,
                                preprocess_graphs=False, 
                                num_process=4,
                                clean_dir=False,                            
                                )
# dataset = TrajectoryDataset(data_dir, phase, skip=8, min_seq_len=2, interaction_rad=3, preprocess=False)
print('{} dataset: {} samples '.format(phase, len(dataset)))
print("Preparing dataloader..")
dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=dataset.collate)
print('Number of %d %s batches (batch size %d) '%(len(dataloader), phase, dataloader.batch_size))
#%%
from utils import *

cfg = Config.fromfile(args.config)
cfg_dict = cfg.serialize

from model.models import gnn_model
from model.losses import *
from model.physics import *

model = gnn_model(cfg.model, cfg.serialize).to(device)
print(model)
# model_attributes(model)
model_parameters(model, verbose=0)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

prediction_model = ConstantVelocity(sec_from_now=np.round(cfg.pred_seq_len * cfg.dt, 3), sampled_at=1/cfg.dt)

# # load model
# src_dir = osp.join(cfg.trajdec_trial, cfg.trajdec_run)
# ckpt_dir = osp.join(src_dir, 'hotel') 
# prediction_model = get_model_from_src(src_dir, ckpt_dir, 'min_test_ade_best', device)[0]

torch.autograd.set_detect_anomaly(True)
phase='train'

start_out = time.time()
if phase=='train':
    model.train()
# pbar = tqdm(total=len(dataloader), position=0)
for iter, (seq_graphs, obsv_graphs, trgt_graphs) in enumerate(dataloader):
    # pbar.update(1)
    start=time.time()
    
    seq_graphs = seq_graphs.to(device)
    obsv_graphs = obsv_graphs.to(device)
    trgt_graphs = trgt_graphs.to(device)

    # remove nodes from trgt_graphs that are not present in obsv graphs
    gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)

    if not(sedges:= sum(gt_graphs.edata['spatial_mask']))>0:
        print('zero spatial edges !!')
    # gt_graphs = remove_node_type(gt_graphs, cfg.node_types)
    # if no_edges([obsv_graphs, trgt_graphs, gt_graphs]):
    if not gt_graphs.number_of_edges()>1:
        continue
    
    # gt_graphs.ndata['pred_pos'] = gt_graphs.ndata['pos']
    # gt_graphs.ndata['pred_hed'] = gt_graphs.ndata['hed']
    # break
    comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))

    gx, xx, ex = prepare_inputs(obsv_graphs, cfg.node_inputs, cfg.edge_inputs)
            
    try:
        # with torch.no_grad():
        if model.__class__.__name__ =='STDec':
            gy, yy, ey = prepare_inputs(gt_graphs, cfg.node_inputs, cfg.edge_inputs)
            
            yy = {k:v.unsqueeze(0) for k, v in yy.items()}
            ey = {k:v.unsqueeze(0) for k, v in ey.items()}
        
            yy['pos'] = gt_graphs.ndata['pos'].unsqueeze(0)
            yy['hed'] = gt_graphs.ndata['hed'].unsqueeze(0)
            yy['speed'] = gt_graphs.ndata['speed'].unsqueeze(0)
            
            if phase=='train':
                logits_n, logits_e = model(gy, y=yy, e=ey, num_samples=1, device=device)
                # pred_dims = {s:loss_func_param(cfg.node_loss, STATE_DIMS[s]) for s in cfg.node_outputs}
                gt_nodes = {s:gt_graphs.ndata[s] for s in cfg.node_outputs}
                gt_edges = {s:gt_graphs.edata[s] for s in cfg.edge_outputs}
                # pred_states = logits_to_dict(logits_n, state_dims=cfg.output_nodes_dim)
                node_loss =  sum([compute_loss(gt_nodes[s], logits_n[s], cfg.node_loss.serialize[s]) for s in cfg.node_outputs])
                loss = node_loss
                if len(cfg.edge_loss.serialize)>0:
                    edge_loss = sum([compute_loss(gt_edges[s], logits_e[s], cfg.edge_loss.serialize[s]) for s in cfg.edge_outputs])
                    loss = node_loss + edge_loss 
                    
            else:
                with torch.no_grad():
                    logits_n, logits_e = model(gy, y=yy, e=ey, num_samples=1, device=device)
                
            # sys.exit()
        else:
            gy, yy, ey = prepare_inputs(seq_graphs, cfg.node_inputs, cfg.edge_inputs)
            if phase=='train':
                logits_y, KLD = model(gx, xx, ex, gy=gy, yy=yy, ey=ey, comm_traj=comm_traj, num_samples=cfg.K, device=device)
                
                _, node_idx, time_idx = traj_to_graph_data(gt_graphs, traj_ids=comm_traj)
                
                logits_n = {s:logits_y[s][..., node_idx, time_idx, :] for s in cfg.node_outputs}# (K, num_nodes, 6)

                gt_nodes = {s:gt_graphs.ndata[s] for s in cfg.node_outputs}
                node_loss =  sum([compute_loss(gt_nodes[s], logits_n[s], cfg.node_loss.serialize[s], mode='average') for s in cfg.node_outputs])
                loss = node_loss + KLD
            else:
                with torch.no_grad():
                    logits_y, logits_n, logits_e, KLD = model(gx, xx, ex, gy=gy, yy=yy, ey=ey, gt=gt, comm_traj=comm_traj, num_samples=cfg.K, device=device)
            
    except Warning as e:
        print(e)
        sys.exit()
        
    # # NOTE! RGCN doesn't support st_dec
    if phase=='train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('ITER:{}, loss: {:.5f}'.format(iter, loss))
        # print('Time required for backward pass:', ((time.time()-start)/(iter+1)))
    break

print('Total time required:', time.time()-start_out)
# outputs =   [loss] #[logits_n[s] for s in cfg.node_outputs]  +
# if len(cfg.edge_outputs)>0:
#     outputs.extend([edge_loss])
    
non_contribs = non_contributing_params(model, [loss])
print('Non contributing params:', non_contribs)
model_parameters(model, verbose=0)
    
# pbar.close()
