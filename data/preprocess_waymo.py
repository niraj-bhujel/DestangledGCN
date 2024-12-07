#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:18:54 2020

@author: niraj
"""

import sys
import os
import math
import time
import random
import scipy
import itertools
import pickle
import numpy as np
import numpy.ma as ma
import multiprocessing as mp
# import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from scipy.interpolate import interp1d
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from glob import glob


import dgl
import torch

from data.lane_utils import *
from data.data_augmentation import *
from data.states import *
from utils.graph_utils import *
from utils.timer import Timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

NUM_GT_RECORDS = {'training':1000, 'validation':150, 'testing': 150}

            
def get_center(all_states, all_states_mask):
  """Gets the center of the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for`all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
  valid_states = all_states[all_states_mask]
  all_y = valid_states[..., 1]
  all_x = valid_states[..., 0]

  center_y = (np.max(all_y) + np.min(all_y)) / 2
  center_x = (np.max(all_x) + np.min(all_x)) / 2

  range_y = np.ptp(all_y)
  range_x = np.ptp(all_x)

  width = max(range_y, range_x)

  return center_y, center_x, width

def interpolate_missing_frames(state_dict, states):
    # Method I, avoid for loops
    timestamp = np.array(state_dict['timestamp_micros'], dtype=np.float)
    timestamp[timestamp==-1] = np.nan 
    
    start_time = np.nanmin(timestamp, axis=-1)            
    end_time = np.nanmax(timestamp, axis=-1)

    pad_front = np.nanargmin(timestamp, axis=-1)
    pad_end = np.nanargmax(timestamp, axis=-1) + 1
    
    seq_duration = np.around((end_time - start_time)/1e6 * 2, 0) + 1
    traj_indices = np.where(seq_duration!=state_dict['valid'].sum(-1))[0]
        
    for n in traj_indices:
        s = pad_front[n]
        e = pad_end[n]
        
        indices = np.where(state_dict['valid'][n]==1)[0]
        
        state_feats = np.stack([state_dict[f][n, indices] for f in states], axis=0) # (num_features, 19)

        interp = interp1d(np.arange(state_feats.shape[-1]), state_feats, axis=1)
        new_feats = interp(np.linspace(0, state_feats.shape[-1]-1, num=e-s))
        
        gt_indices = np.arange(s, e)
        missing_frames = np.setxor1d(indices, gt_indices) # frame that is in gt_indices but not in indices, rangess from 0-19
        
        for i, f in enumerate(states):
            state_dict[f][n][missing_frames] = new_feats[i][missing_frames-s]
            
    return state_dict

def compute_pad(timestamp):
    timestamp = np.array(timestamp, dtype=np.float)
    timestamp[timestamp==-1] = np.nan 

    pad_front, pad_end = np.nanargmin(timestamp, axis=-1), np.nanargmax(timestamp, axis=-1) + 1
    
    return pad_front, pad_end
                
def compute_padded_diff(array, mask, end_idxs):
    '''
    array: (n, 19, 2)
    mask: (n, 19)
    end_idx: (n, ) # last valid index 
    Main idea is to extend a column at second axis, and copy the pad_end value to mimick repitition of last value
    '''
    
    # compute velocity at once
    row_indices = np.arange(array.shape[0])
    mask = np.expand_dims(mask, axis=-1)
    diff = np.diff(array * mask, axis=1) # (n, 18, 2)
    
    # repeat last column of second axis 
    pad_diff = np.concatenate([diff, np.full((diff.shape[0], 1, 2), fill_value=-1)], axis=1) # extend the second axis (n, 19, 2)
    
    # copy pad_end value from diff to pad_diff
    pad_diff[row_indices, end_idxs] = diff[row_indices, end_idxs-1]
    
    return pad_diff
    
def feature_to_sequence_dict(state_dict, dt, max_vel=80, max_acc=20):
    state_dict = state_dict.copy()
    
    # num_agents = len(state_dict['id'])
    # valid_indices = [np.where(valid==1)[0] for valid in state_dict['valid']]
    # state_dict['vx'] = np.full_like(state_dict['velocity_x'], fill_value=-1.)
    # state_dict['vy'] = np.full_like(state_dict['velocity_y'], fill_value=-1.)
    # state_dict['ax'] = np.full_like(state_dict['velocity_x'], fill_value=-1.)
    # state_dict['ay'] = np.full_like(state_dict['velocity_y'], fill_value=-1.)
    # for i in range(num_agents):
    #     idxs = valid_indices[i]
    #     px = state_dict['x'][i][idxs]
    #     py = state_dict['y'][i][idxs]
    #     # dx = np.diff(px)
    #     # dy = np.diff(py)
    #     vx = np.gradient(px, dt)
    #     vy = np.gradient(py, dt)
    #     ax = np.gradient(vx, dt)
    #     ay = np.gradient(vy, dt)
    #     state_dict['vx'][i][idxs] = vx
    #     state_dict['vy'][i][idxs] = vy
    #     state_dict['ax'][i][idxs] = ax
    #     state_dict['ay'][i][idxs] = ay
        
    pad_front, pad_end = compute_pad(state_dict['timestamp_micros'])

    pos = np.stack([state_dict['x'], state_dict['y']], axis=-1)
    
    for f in ['x', 'y']:
        row_indices = np.arange(state_dict[f].shape[0])
        end_val = state_dict[f][row_indices, pad_end-1]
        rel_val = state_dict[f] - end_val.reshape(-1, 1)
        state_dict[f'rel_{f}'] =  np.where(state_dict['valid']==0, -1, rel_val)
        
        
    vel = compute_padded_diff(pos, mask=state_dict['valid'], end_idxs=pad_end-1)/dt
    # vel = np.stack([state_dict['vx'], state_dict['vy']], axis=-1)
    # vel = np.stack([state_dict['velocity_x'], state_dict['velocity_y']], axis=-1)
    # vel = np.clip(vel, -max_vel, max_vel) # clip here before computing vnorm and heading

    vnorm = np.linalg.norm(vel, axis=-1, keepdims=True)
    vdir = np.divide(vel, vnorm, out=np.zeros_like(vel), where=(vnorm > 0.))
    
    rel = np.stack((state_dict['rel_x'], state_dict['rel_y']), axis=-1)
    acc = compute_padded_diff(vel, mask=state_dict['valid'], end_idxs=pad_end-1)/dt
    # acc = np.stack([state_dict['ax'], state_dict['ay']], axis=-1)
    # acc = np.gradient(vel, dt, axis=2)
    # acc = np.clip(acc, -max_acc, max_acc)
    
    anorm = np.linalg.norm(acc, axis=-1, keepdims=True)
    
    yaw = np.expand_dims(state_dict['bbox_yaw'], 2)
    
    sequence_dict = {}
    
    sequence_dict['pos'] = pos
    sequence_dict['rel'] = rel
    sequence_dict['vel'] = vel
    sequence_dict['vnorm'] = vnorm
    
    sequence_dict['acc'] = acc
    sequence_dict['anorm'] = anorm
    
    sequence_dict['dir'] = vdir
    sequence_dict['yaw'] = yaw
    
    sequence_dict['msk'] = state_dict['valid']>0
    sequence_dict['fid'] = state_dict['timestamp_micros']

    sequence_dict['nid'] = state_dict['id']
    sequence_dict['cid'] = state_dict['type']-1 # waymo type start from 1 
            
    return sequence_dict

def _parse(value):
    return tf.io.parse_single_example(value, features_description)


#%%
def preprocess_sequence(data_dir, version, phase, dt=0.5, obsv_len=3, 
                        seq_len=19, min_obsv_len=2, min_seq_len=2, skip=0, min_agents=1, 
                        include_lane=False, visualize_traj=False, visualize_graph=False, 
                        save_path=None, pid=0, tf_records=None): 

    # print(f'Preprocessing ... \npid:{pid} \nphase:{phase} \ndt:{dt} \nmin_seq_len:{min_seq_len}, \nskip:{skip}, \nmin_agents:{min_agents}')
#%%

    info = ['id', 'type', 'is_sdc', 'tracks_to_predict']
    states = ['x', 'y', 'velocity_x', 'velocity_y', 'bbox_yaw', 'timestamp_micros', 'valid'] 

    if tf_records is None:
            
        tf_records = sorted(glob(os.path.join(data_dir, phase, "*")))

    print('*' * 25, f"START: PID-{pid}", '*' * 25)
    print(f'[INFO] PID-{pid}: Total sequences: {len(tf_records)}')

    if not len(tf_records)>0:
        raise Exception(f'Data dir {os.path.join(data_dir, phase)} is empty')

    global_tid=0
    total_seq = 0

    processed_seqs = []
        
    min_seq_len = 1 if phase=='testing' else min_seq_len
    
    start = time.time()
    
    for record in tf_records: # pid must start from zero
        
        # record_id = record_name.split('/')[-1].split('.')[-1].split('-')[1]
        record_id = record.split('/')[-1]
        print(f"[INFO] PID-{pid}: Processing RECORD {record_id}/{len(tf_records)}")
        
        sample_lists = [os.path.join(record, sample) for sample in os.listdir(record)]
        
        timer = Timer()
        timer.tic(tic_once=True)
        
        for idx, sample in enumerate(sample_lists):    

            with open(sample, 'rb') as f:
                feature_dict = pickle.load(f)
            
            state_dict = feature_dict['state']
            traffic_lights = feature_dict['traffic']
            
            # sampling
            sampling_indices = np.arange(0, 91, 5) # new sampling rate will be 2Hz i.e. 0.5 secs
            for f in states:
                state_dict[f] = state_dict[f][:, sampling_indices]
            
            for k in traffic_lights.keys():
                traffic_lights[k] = traffic_lights[k][sampling_indices[:3]]
            
            if phase!='testing':

                # deal with seq len
                valid_indices = np.sum(state_dict['valid'], -1) > int(min_seq_len) # 2 hz
                
                # deal with static agents
                vel = np.stack([state_dict['velocity_x'], state_dict['velocity_y']], axis=-1)
                vel_norm = np.linalg.norm(vel * state_dict['valid'][..., np.newaxis], axis=-1)
                valid_indices *= vel_norm.max(-1)>0.01
                
                # vel_norm = np.ma.masked_array(vel_norm, mask=state_dict['valid']==0) # mask invalid values i.e. 0
                # valid_indices *= vel_norm.max(-1).data<80.
                
                # skip traj with missing frame
                # timestamp = state_dict['timestamp_micros']
                # pad_front, pad_end = compute_pad(timestamp)
                
                start_time = np.nanmin(state_dict['timestamp_micros'], axis=-1)
                end_time  = np.nanmax(state_dict['timestamp_micros'], axis=-1)           
                seq_duration = np.around((end_time - start_time)/1e6 * 2.0, 0) + 1 # 2.0 -> 1/0.5
                
                missing_traj_indices = np.where(seq_duration!=state_dict['valid'].sum(-1))[0]
                
                valid_indices *= np.logical_not(np.isin(np.arange(len(state_dict['x'])), missing_traj_indices))
                
                # # filter
                # for f in info + states:
                #     state_dict[f] = state_dict[f][valid_indices]

                # check outliers in pos: perform masked operation due -1 value
                pos = np.stack([state_dict['x'], state_dict['y']], axis=-1)
                pos_ma = np.ma.masked_array(pos, mask=np.tile(state_dict['valid'][:, :, np.newaxis], (1, 1, 2))==0)
                pos_ma_dnorm = np.sqrt(np.sum(np.ma.diff(pos_ma, axis=1)**2, axis=-1))
                valid_indices *= pos_ma_dnorm.std(-1).data<5.0
            
                for f in info + states:
                    state_dict[f] = state_dict[f][valid_indices]
                

            # timer.toc() # 4400 fps 
            num_agents = len(state_dict['id'])
            # print(num_agents)
            
            if num_agents > min_agents-1:                

                sequence_dict = feature_to_sequence_dict(state_dict, dt) # sample from 91 to 19 
                
                # print(sequence_dict['vnorm'])
                if sequence_dict['vnorm'].max()>80:
                    print('vnorm > 80')
                    # sys.exit()

                center_y, center_x, width = get_center(sequence_dict['pos'], sequence_dict['msk'])
                sequence_dict['center'] = np.tile([center_x, center_y], (num_agents, 1))

                sequence_dict['sid'] = np.full((num_agents, ), fill_value=int(record_id))

                sequence_dict['tid'] = global_tid + np.arange(num_agents)
                global_tid += num_agents

                # print(f"[INFO] PID-{pid}:", sequence_dict['tid'])

                if save_path is not None:
                    with open(f'{save_path}/seq_graph_record{record_id}-{idx:05d}.bin', 'wb') as f:
                        pickle.dump(sequence_dict, f)
                else:
                    processed_seqs.append(sequence_dict)

                total_seq += 1

                if visualize_traj:
                    comm_traj = np.sort(np.intersect1d(obsv_graph.ndata['tid'].cpu().numpy(), trgt_graph.ndata['tid'].cpu().numpy()))
                    obsv_pos = [obsv_graph.ndata['pos'][obsv_graph.ndata['tid']==tid] for tid in comm_traj]
                    trgt_pos = [trgt_graph.ndata['pos'][trgt_graph.ndata['tid']==tid] for tid in comm_traj]
                    fig, ax = plot_path(obsv_pos, trgt_pos)
                    fig.savefig(f'../../vis_traj/preprocessing/waymo/{phase}_{record_id}_frame{idx}.png', dpi=300, bbox_inches='tight')
                
                if visualize_graph:
                    plt.close('all')
                    obsv_graph, trgt_graph = split_graph(g, split_idx=obsv_len-1)
                    fig, ax = network_draw(dgl.batch([obsv_graph, trgt_graph]) , node_label='cid', node_size=100, figsize=(8,8), 
                                            fontsize=6, pad=10, axis_off=False, limit_axes=False)
                    if include_lane:
                        ax.add_collection(LineCollection([line for _, line in scene_lane_poses.items()], color='#808080')) # curr lane poses
                        for nid, node_lanes in curr_node_lanes.items():
                            ax.add_collection(LineCollection([scene_lane_poses[l] for l in node_lanes], color=get_color(nid, line_colors)))
                    fig.savefig(f'../../vis_graphs/preprocessing/waymo/{phase}_{record_id}_frame{idx}.png', dpi=300, bbox_inches='tight')
            
            timer.toc() # 60 fps
            # break

        # print(f'[INFO] PID-{pid}:Time/record: {timer.total_time:.03f} fps: {timer.fps:.03f} sequences: {total_seq}')
        break

    #%%
    # pbar.close()
    # print(f'[INFO] PID-{pid}: Finished processing {total_seq} {phase} sequences in {time.time()-start:.1f}s')
    print('*' * 25, f"END: PID-{pid}", '*' * 25)
    return processed_seqs
    
    
#%%
if __name__=='__main__':
    
    import sys
    import os
    
    import numpy as np
    import tensorflow as tf
    
    import warnings
    warnings.filterwarnings("ignore")
    
    src_path = '/home/dl-asoro/Desktop/Recon_GCN/src'
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # print(sys.path)
    
    from data.states import NODE_TYPES, INTERACTION_RADIUS
    
    # from visualization.trajectory_visualization import plot_path
    # from visualization.vis_graph import network_draw, line_colors, get_color
    # from utils.graph_utils import split_graph, remove_redundant_nodes, seq_to_graph
    # from utils.timer import Timer
    from argument_parser import parse_args

    args = parse_args()
    
    # physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([], 'GPU')

    obsv_len = 3
    seq_len = 19

    min_obsv_len = 1
    min_seq_len = 2
    min_agents = 1
    
    skip = 0
    dt = 0.5
    
    include_lane=False
    
    version='full_10'
    
    phase = 'training'
    
    visualize_graph = False
    visualize_traj = False
    interaction_radius = INTERACTION_RADIUS
    aug_prob = 0.5
    
    pid = 0
    save_path = None
    
    data_dir = '/run/user/1000/gvfs/smb-share:server=access.serc.acrces2.shared-svc.local,share=i2r/home/bhujeln1/waymo-od/tf_example'
    # data_dir = '/home/niraj/Desktop/waymo-od/uncompressed/tf_example'
    # data_dir = '../datasets/waymo'
    # data_dir = '/mnt/local_share/waymo-od/tf_example'
    # data_dir = '/mnt/dl-asoro/datasets/waymo/'
    # data_dir = '/media/iirav/hdd2/waymo'
    data_dir = '/home/dl-asoro/Desktop/Recon_GCN/datasets/waymo/processed_data'
