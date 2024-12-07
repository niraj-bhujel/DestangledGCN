#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:44:40 2023

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

from data.dataset import PedestriansDataset, NuScenesDataset, WaymoDataset
from data.states import *

from utils.config import Config
from utils.graph_utils import *
from utils.train_utils import *

from visualization.vis_graph import *
from visualization.vis_traj import *

def rotate_pos(pos, cx=0, cy=0, rotation=0):
    '''
    p: L x 2
    T: tuple(x, y)
    alpha: rotation angle in degree
    '''
    T = np.array([[cx, cy]])
    alpha = rotation * math.pi / 180
    M = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    return np.dot(pos - T, M) + T 



if __name__=='__main__':
    
    config = "./configs/pedestrian_gcn.py"
    
    cfg = Config.fromfile(config)
    cfg_dict = cfg.serialize
    # print({k:v for k, v in cfg.serialize.items() if k!='standardization'})

    torch.cuda.init()
    device = torch.device('cuda:0')
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(cfg.seed)


    all_dsets = {'pedestrians':PedestriansDataset, 'nuscenes': NuScenesDataset, 'waymo': WaymoDataset}
    cfg.node_types = ['PEDESTRIAN']
    cfg.include_lane = False
    
    
    version = ''
    # version = 'mini'
    # version = 'full'
    dset_name = 'hotel'
    phase = 'test'
    
    # create dataset
    dataset = PedestriansDataset(f'../../datasets/pedestrians/{dset_name}/',
                                phase=phase,
                                version=cfg.version,
                                node_types=cfg.node_types,
                                interaction_radius=INTERACTION_RADIUS,
                                obsv_len=cfg.obsv_seq_len,
                                pred_len=cfg.pred_seq_len,
                                min_obsv_len = cfg.min_obsv_len,
                                min_seq_len=cfg.min_seq_len,
                                min_agents=cfg.min_agents,
                                skip=0,
                                aug_prob=0, # augment only train data
                                dt=cfg.dt,
                                fov_graph=False,
                                preprocess=False,
                                )
    # dataset = TrajectoryDataset(data_dir, phase, skip=8, min_seq_len=2, interaction_rad=3, preprocess=False)
    print('{} dataset: {} samples '.format(phase, len(dataset)))

    print("Preparing dataloader..")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate)
    print('%d %s batches (batch size %d) '%(len(dataloader), phase, dataloader.batch_size))
    
    #%%
    from visualization.vis_graph import *


    
    BACKGROUND_FRAME = True

    plot_dir = f"../vis_spatial_graphs/{dset_name}_{phase}_background{BACKGROUND_FRAME}"
    os.makedirs(plot_dir, exist_ok=True)
    
    for iter, (seq_graphs, obsv_graphs, trgt_graphs) in enumerate(dataloader):
        
        # if not iter>420:
        #     continue
        
        if no_edges([obsv_graphs]):
            print('NO EDGES')
            continue
        
        scene_num = obsv_graphs.ndata['sid'].unique().item()
        frame_num = int(obsv_graphs.ndata['fid'].unique().max())

        try:
            frame = plt.imread(f"{ROOT_DIR}/eth_ucy_frames/{dset_name}/frames/frame{frame_num}.png") if BACKGROUND_FRAME else None
        except Exception:
            frame = None
            # continue
        
        # break
        
        padding = {'eth': (0, 1, 3, 1), 'hotel': (3.5, 3.5, -2, -1), 'univ': (1, 1, 1, 1), 
                   'zara1': (0, 0, 0, 0), 'zara2': (0.5, 1, -0.5, 0),}
        
        rotations = {'eth':90, 'hotel':89, 'univ':-90, 'zara1':0, 'zara2':0}
    
        if rotations[dset_name]!=0:
            cx = (data_stats[dset_name]['x_max'] - abs(data_stats[dset_name]['x_min']))/2
            cy = (data_stats[dset_name]['y_max'] - abs(data_stats[dset_name]['y_min']))/2
            
        if rotations[dset_name]!=0:
            
            # compute new vel
            for tid in obsv_graphs.ndata['tid'].unique():
        
                tid_idx = obsv_graphs.ndata['tid']==tid
                
                p = obsv_graphs.ndata['pos'][tid_idx].numpy()
                p = rotate_pos(p, cx, cy, rotations[dset_name])

                v = np.gradient(p, cfg.dt, axis=0)
                v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
                
                v_dir = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 0.))
                
                obsv_graphs.ndata['pos'][tid_idx] = torch.tensor(p, dtype=torch.float)
                obsv_graphs.ndata['vel'][tid_idx] = torch.tensor(v, dtype=torch.float)
                obsv_graphs.ndata['vnorm'][tid_idx] = torch.tensor(v_norm, dtype=torch.float)
                obsv_graphs.ndata['dir'][tid_idx] = torch.tensor(v_dir, dtype=torch.float)

        sub_nodes = obsv_graphs.nodes()[obsv_graphs.ndata['ntx'].flatten()==7]
        
        g = obsv_graphs.subgraph(sub_nodes)
        
        if no_edges([g]):
            print('NO EDGES!!')
            continue
        
        network_draw(g, show_node_label=False, node_label='nid', show_edge_labels=False, edge_label='dist', 
                     show_legend=True, node_size=1500, show_direction=True, frame=frame, 
                     extent=data_stats[dset_name], pad=padding[dset_name],  limit_axes=True, 
                     counter=iter, save_dir=plot_dir, dtext=f"Frame_{frame_num:06d}", axis_off=True,
                     directioncolor='lawngreen')
        
        
        # if iter>500:
        #     break
    
        # break