#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:54:28 2021

@author: dl-asoro
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

import dgl
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = '/'.join(ROOT_DIR.split('/')[:-2])

print('ROOT_DIR', ROOT_DIR)
os.environ['ROOT_DIR'] = ROOT_DIR

src_dir = ROOT_DIR + '/src'
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# os.chdir(os.path.dirname(os.path.realpath(__file__)))


from utils.config import Config

torch.cuda.init()
device = torch.device('cuda:0')

# config = './configs/base/pedestrians.py'
config = './configs/pedestrian_stdec_sf.py'

# dataset_config = './configs/pedestrians/gcn.py'
# net_config = './configs/nets/gcn_config.py'

# config = './configs/waymo/gcn_config.py'
# config = './configs/waymo/sa_gcn_config.py'
# config = './configs/waymo/san_config.py'
# config = './configs/waymo/stdec_gcn_config.py'

cfg = Config.fromfile(config)
cfg_dict = cfg.serialize
# print({k:v for k, v in cfg.serialize.items() if k!='standardization'})

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if device.type=='cuda':
    torch.cuda.manual_seed_all(cfg.seed)
    
#%%
from data.dataset import PedestriansDataset, NuScenesDataset, WaymoDataset
from data.preprocess_utils import *
from utils.graph_utils import *
from utils.train_utils import *

all_dsets = {'pedestrians':PedestriansDataset, 'nuscenes': NuScenesDataset, 'waymo': WaymoDataset}
cfg.node_types = ['PEDESTRIAN']
cfg.include_lane = False


version = ''
# version = 'mini'
# version = 'full'
# data_dir = '../../datasets/nuscenes/processed_data'
data_dir = '../datasets/pedestrians/eth/'
phase = 'test'
# shuffle = True if phase=='train' else False
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
                                skip=cfg.skip if phase=='train' else 1,
                                aug_prob=0,
                                dt = cfg.dt,
                                min_agents = cfg.min_agents,
                                # include_robot=cfg.include_robot,
                                include_lane=cfg.include_lane,
                                # pos_enc=cfg.pos_enc,
                                # pos_enc_dim=cfg.pos_enc_dim,
                                full_graph = False,
                                preprocess=False,
                                preprocess_graphs=False, 
                                num_process=4,
                                clean_dir=False,
                                
                                )
# dataset = TrajectoryDataset(data_dir, phase, skip=8, min_seq_len=2, interaction_rad=3, preprocess=False)
print('{} dataset: {} samples '.format(phase, len(dataset)))
#%%
from visualization.vis_graph import *


    
spatial_edges = []
temporal_edges = []
print("Preparing dataloader..")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate)
print('%d %s batches (batch size %d) '%(len(dataloader), phase, dataloader.batch_size))
    
start = time.time()
total_traj = []
total_peds = []
cfg.input_states = ['pos']
cfg.input_edges = ['dist']

tid_list = []
for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):

    obsv_graphs = obsv_graphs.to('cuda:0')
    trgt_graphs = trgt_graphs.to('cuda:0')
    
    # remove nodes from trgt_graphs that are not present in obsv graphs
    gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)
    
    if no_edges([obsv_graphs, trgt_graphs, gt_graphs]):
        print('NO EDGES')
        continue

    # common traj in obsv and gt
    comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))
    
    if not len(comm_traj)>3:
        continue
    
    # prepare inputs/standardize   
    gx, xx, ex = prepare_inputs(obsv_graphs, cfg.input_states, cfg.input_edges,)
    gy, yy, ey = prepare_inputs(gt_graphs, cfg.input_states, cfg.input_edges)
    g = gt_graphs
    
    # network_draw(dgl.batch([obsv_graphs, trgt_graphs]), show_node_labels=True, node_label='ntx')
    # print('ITER-{}/{}:{:.6f} seconds/iter'.format(iter, len(dataloader), (time.time()-start)/(iter+1)))
    
    # assert (gt_graphs.ndata['cid']==2).sum() == (gt_graphs.ndata['cid']==0).sum()
    # total_traj.append(len(gt_graphs.ndata['tid'].unique()))
    # total_peds.append(gt_graphs.ndata['nid'].unique().tolist())
    # print(gt_graphs.ndata['cid'].unique())
    # traj_len = [(gt_graphs.ndata['tid']==tid).sum() for tid in comm_traj]
    # print(traj_len)
    # gx, xx, ex = prepare_inputs(obsv_graphs, cfg.input_states, cfg.input_edges)
    
    # if xx.isnan().any() or ex.isnan().any():
    #     raise Exception ('nan')
    # print(gx.ndata['sid'].unique())
    # for obsv_graph, gt_graph in zip(dgl.unbatch(obsv_graphs), dgl.unbatch(gt_graphs)):
    #     new_comm_traj = np.sort(np.intersect1d(obsv_graph.ndata['tid'].cpu().numpy(), gt_graph.ndata['tid'].cpu().numpy()))
    #     obsv_traj = [obsv_graph.ndata['pos'][obsv_graph.ndata['tid']==tid].cpu().numpy() for tid in new_comm_traj]
    #     gt_traj = [gt_graph.ndata['pos'][gt_graph.ndata['tid']==tid].cpu().numpy() for tid in new_comm_traj]
    #     # pred_traj = [raw_data['pred'][:, i, :, :].cpu().numpy() for i, tid in enumerate(raw_data['comm_traj']) if tid in new_comm_traj]
    #     pred_traj = [traj[None, :, :] for traj in gt_traj]
    #     scene_id = obsv_graph.ndata['sid'].unique().item()
    #     plot_kde(obsv_traj, gt_traj, pred_traj, save_dir='../../plot_traj/' + 'val/', counter='scene_%d'%scene_id)
    #     break

    # if len(comm_traj)>5:
    #     break
    break

    spatial_edges.append(sum(gt_graphs.edata['spatial_mask'].cpu().numpy()==1))
    temporal_edges.append(sum(gt_graphs.edata['spatial_mask'].cpu().numpy()==0))
    
    total_peds.append(gt_graphs.ndata['nid'].unique().tolist())
    
    tid_list.append(dgl.batch([gx, gy]).ndata['tid'].unique())
print('Total traj:', sum(total_traj))
print('Total Ped:', len(np.unique(total_peds)))
print('Total time:', time.time()-start)
# print('total-tid', len(torch.cat(tid_list)))
# print('unique-tid', len(torch.cat(tid_list).unique()))

print('Total spatial edges:', np.sum(spatial_edges))
print('Total temporal edges:', np.sum(temporal_edges))

# plt.close('all')
#%%
# from visualization.vis_graph import network_draw
# from visualization.vis_traj import plot_kde, plot_path

# # trgt_sub_graphs = [trgt_graphs.subgraph(gt_graphs.nodes()[gt_graphs.ndata['nid']==nid]) for nid in gt_graphs.ndata['nid'].unique()]
# # fig = network_draw(trgt_sub_graphs[i], show_node_labels=True, node_label='cid', show_legend=True, font_size=8, fig_name='gt')
# network_draw(gt_graphs, show_node_labels=True, node_label='tid', show_legend=False, font_size=8, fig_name='gt', 
#               pad=20, axis_off=True, limit_axes=True,)



