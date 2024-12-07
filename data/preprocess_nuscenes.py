#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:18:54 2020

@author: niraj
"""

import sys
import os
import dill
import math
import time
import random
import scipy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import dgl
import torch

from data.states import *
from data.preprocess_utils import *
from data.lane_utils import *

def preprocess_sequence(data_dir, phase, version, obsv_len=8, seq_len=20, min_obsv_len=2, 
                        min_seq_len=2, skip=0, min_ped=1, include_robot=False, include_lane=False, 
                        dt=0.5, node_type='PEDESTRIAN', visualization=False):  

    print("Preprocessing {} {} {} {} sequence ... ".format(version, phase, node_type, 'lane' if include_lane else ''))
    
    nu_path = '../../nuscenes-devkit/python-sdk'
    if nu_path not in sys.path:
        sys.path.insert(0, nu_path)

    from nuscenes.map_expansion.arcline_path_utils import discretize_lane

#%%    
    mini_string = ''
    if 'mini' in version:
        mini_string = '_mini'
    data_dict_path = os.path.join(data_dir, '/nuScenes_' + phase + mini_string + '_full.pkl')
    with open(data_dict_path, 'rb') as f:
        scenes_data, scenes_map = dill.load(f)
    
    scenes = list(scenes_data.keys())
    # use only 40% for faster training, at least one scene
    if 'semi' in version:
        random.seed(42)
        scenes = random.sample(scenes, max([1, int(40/100 * len(scenes))]) ) 
    
#%%    
    pred_len = seq_len - obsv_len
    token_map = {}
    processed_sequence = []

    global_tid = 0
    scene_pos_mean = {}
    
    pbar = tqdm(total=len(scenes), position=0) 
    start = time.time()
    for i, scene in enumerate(scenes):
        pbar.update(1)
        print('SCENE: {}/{}'.format(i, len(scenes)))

        scene_id = int(scene.split('-')[1])
        
        # if scene_id!=103:
        #     continue
    
        # if scene_id!=916:
        #     continue
        # if scene_id != 61:
        #     continue
        
        data = scenes_data[scene]
        data = data[data['type']==node_type]
        
        x_mean, y_mean = data['x'].mean(), data['y'].mean()
        scene_pos_mean[scene_id] = [x_mean, y_mean]
        
        if include_lane:
            nusc_map = scenes_map[data['map_name'].iloc[0]]
            patch_box = (data['x'].min(), data['y'].min(), data['x'].max(), data['y'].max())
            lane_records = nusc_map.get_records_in_patch(patch_box, ['lane', 'lane_connector'])
            # lane_records = lane_records['lane'] + lane_records['lane_connector']

            scene_lane_poses = dict()
            for lane_token in lane_records['lane'] + lane_records['lane_connector']:
                my_lane = nusc_map.arcline_path_3.get(lane_token, [])
                discretized = np.array(discretize_lane(my_lane, resolution_meters=0.5))
                scene_lane_poses[lane_token] = discretized[:, :2]
        
        node_ids_dict = {hex_id:int_id for hex_id, int_id in zip(data['node_id'], preprocessing.LabelEncoder().fit_transform(data['node_id']))}


        frames = pd.unique(data['frame_id']).tolist()
        frame_data = [data[data['frame_id'] == frame] for frame in frames]

        for idx in range(0, len(frames), skip+1):
            # print("SCENE:{}, FRAME:{}".format(scene_id, idx))    
            
            # if idx!=11:
            #     continue
        
            start_idx = max(0, idx-obsv_len+1)
            end_idx = min(len(frames), idx+pred_len+1)         
            # print(idx, start_idx, end_idx, end_idx-start_idx, len(frame_data[start_idx:end_idx]))

            curr_seq = pd.concat(frame_data[start_idx:end_idx])
            # curr_seq = pd.concat(frame_data[idx:idx + seq_len], axis=0)
            
            if include_lane:
                curr_lane_poses = trim_lanes_poses(scene_lane_poses, curr_seq[['x', 'y']].values.min(0), curr_seq[['x', 'y']].values.max(0))
            
            curr_nodes_pos = []
            curr_node_lanes = {}
            num_nodes_considered = 0
            curr_sequence = defaultdict(list)
            
            current_nodes = pd.unique(frame_data[idx]['node_id'])
            for n, node_id in enumerate(current_nodes):

                node_data = curr_seq[curr_seq['node_id']==node_id]

                nid = node_ids_dict[node_data['node_id'].iloc[0]]
                
                # if not int(nid)==16:
                #     continue
            
                # if (scene_id!=103) or (idx!=8) or nid!=48:
                #     continue
                
                # seq less than 2 will have problem computing gradients
                if not len(node_data)>1:
                    continue

                if node_id=='ego':
                    if not include_robot:
                        continue
                
                node_frames = node_data['frame_id'].values
                
                # index of current frame_num in node data
                node_idx = list(node_frames).index(frames[idx]) 
                
                node_history_len = node_idx + 1 # node history include current index as well
                node_future_len = len(node_frames) - node_idx - 1 
                
                assert node_history_len + node_future_len == len(node_frames)
                
                # pad_front and pad front are index value
                pad_front = obsv_len - node_history_len
                pad_end = obsv_len + node_future_len

                # at least min_obsv_len, pad_front can range from 0 - 19
                if not obsv_len-pad_front>min_obsv_len-1:
                    continue
                
                # at least one gt future pos, avoid error during graph split, but this will affect past encoder
                if not pad_end-obsv_len>0:
                    continue
                
                # at least sequence length of min_seq_len
                if not pad_end-pad_front>min_seq_len-1:
                    continue
                
                node_seq = node_data[['x', 'y']].values

                curvature, path_len, path_dist, = trajectory_curvature(node_seq)
                print(curvature)
                
                if path_dist<0.9:
                    # print('static vehicle')
                    continue
                    # node_seq = np.tile(node_seq[0], (len(node_seq), 1))
                node_states = node_sequence(node_seq, 
                                            node_data['ts'].values, 
                                            pad_front, 
                                            pad_end, 
                                            seq_len, 
                                            rel_idx=-1,
                                            frames=node_data['frame_id'].values,
                                            yaw=node_data['yaw'].values)

                for state in node_states.keys():
                    curr_sequence[state].append(node_states[state])
                
                # curr_sequence['msk'].append(node_msk)
                # curr_sequence['fid'].append(node_fid)

                curr_sequence['nid'].append(nid)
                curr_sequence['tid'].append(global_tid)
                curr_sequence['cid'].append(NODE_TYPES.index(node_type))
                curr_sequence['sid'].append(scene_id)
                
                instance = node_data['instance_token'].values[node_idx]
                sample = node_data['sample_token'].values[node_idx]
                
                token_map[global_tid] = '_'.join([instance, sample])
                
                # nid+=1
                global_tid+=1
                num_nodes_considered+=1
                
                curr_nodes_pos.append(node_seq)
                # curr_nodes_pos.append(node_seq[:obsv_len-pad_front])
                
                if include_lane and node_future_len>0:
                    node_history = node_seq[:obsv_len-pad_front]

                    start_lanes, nearest_lane = nearest_node_lanes(node_history, curr_lane_poses, 
                                                                   max_angle=60 if curvature>0 else 45, 
                                                                   trim_lanes=True, 
                                                                   d_thresh=3)
                    # sys.exit()
                    
                    if not len(start_lanes)>0:
                        continue

                    curr_node_lanes[nid] = []
                    for start_lane in start_lanes[:2]:
                        node_lanes, paths = lanes_along_node(start_lane, node_history, curr_lane_poses, nusc_map, radius=20, max_dist=100)
                        curr_node_lanes[nid].extend(node_lanes)
                        

                        for path in paths:
                            
                            path = [lane for lane in path if lane is not None]
                            
                            if not len(path)>0:
                                continue
    
                            lane_pos = np.concatenate([curr_lane_poses[lane] for lane in path])               
    
                            # METHOD I: sample lane points based on gt pos
                            #lane_points = closest_lane_points(lane_pos, node_seq)
                            
                            # METHOD II: sample lane points based on cv future 
                            lane_points = closest_lane_points(lane_pos, np.concatenate([node_history, node_cv_future]))
                            
                            # METHODIII: sample lane pos without future node pos 
                            # closest_obsv_idxs = euclidean_distances(lane_pos, node_history).argmin(0) # closest point to start pos 
                            # closest_trgt_start_idx = closest_obsv_idxs[-1] + (closest_obsv_idxs[-1] - closest_obsv_idxs[-2])
                            # obsv_lane_pos = lane_pos[closest_obsv_idxs]
                            # trgt_lane_pos = lane_pos[closest_trgt_start_idx:]
                            # if not len(trgt_lane_pos)>0:
                            #     continue
                            # if len(trgt_lane_pos)>12:
                            #     trgt_lane_pos = sample_lane_points(trgt_lane_pos, num_points=12)
                            # pad_end = obsv_len+len(trgt_lane_pos)
                            # lane_points = np.concatenate([obsv_lane_pos, trgt_lane_pos])
    
                            #TODO, orientation should be array
                            lane_states, lane_fid, lane_msk = node_sequence(lane_points, dt, 
                                                                            pad_front=pad_front, 
                                                                            pad_end=pad_end,
                                                                            seq_len=seq_len, 
                                                                            rel_idx=-1,
                                                                            # frames=curr_nodes_fid[i], 
                                                                            orientation = None)
                            
                            for state in lane_states.keys():
                                curr_sequence[state].append(lane_states[state])
                            
                            # curr_sequence['fid'].append(lane_fid)
                            curr_sequence['msk'].append(lane_msk)
                            curr_sequence['nid'].append(nid)
                            curr_sequence['tid'].append(global_tid)
                            curr_sequence['sid'].append(scene_id)
                            curr_sequence['cid'].append(NODE_TYPES.index('LANE'))
                                                
                            global_tid += 1
                            # break

            if num_nodes_considered > min_ped-1:
                
                for key, _ in curr_sequence.items():
                    curr_sequence[key] = np.stack(curr_sequence[key] , 0)
                
                if visualization:
                    plt.close('all')
                    obsv_graphs, trgt_graphs = split_graph(seq_to_graph(deepcopy(curr_sequence), INTERACTION_RADIUS))
                    fig, ax = network_draw(dgl.batch([obsv_graphs, trgt_graphs]) , node_label='nid', node_size=100, figsize=(8,8), 
                                           fontsize=6, pad=10, axis_off=False, limit_axes=False)
                    ax.add_collection(LineCollection([line for _, line in scene_lane_poses.items()], color='#808080')) # curr lane poses
                    for nid, node_lanes in curr_node_lanes.items():
                        ax.add_collection(LineCollection([scene_lane_poses[l] for l in node_lanes], color=get_color(nid, line_colors)))
                    fig.savefig('../../vis_graphs/preprocessing/{}_scene{}_frame{}.png'.format(phase, scene_id, idx), dpi=300, bbox_inches='tight')

                # standardize pos 
                curr_sequence['pos'] = curr_sequence['pos'] - np.array([x_mean, y_mean])[None, None, :]
                processed_sequence.append(curr_sequence)
                
                # break
        # break
    
    #%%
    pbar.close()
    print('\nFinished preprocessing {} {} sequences in {:.1f}s'.format(len(processed_sequence), phase, time.time()-start))
    print('Total number of {} trajectories:'.format(phase), global_tid)
    
    return processed_sequence, token_map, scene_pos_mean
#%%
if __name__=='__main__':

    import warnings
    warnings.filterwarnings("ignore")

    if './' not in sys.path:
        sys.path.insert(0, './')
    
    # print(sys.path)

    from data.utils import NODE_TYPES, INTERACTION_RADIUS
    from visualization.vis_graph import network_draw, line_colors, get_color
    from utils.graph_utils import seq_to_graph, split_graph, remove_redundant_nodes
    
    
    obsv_len = 8
    seq_len=20

    min_obsv_len=2
    min_seq_len=2
    min_ped=1
    
    skip=1
    dt=0.5
    
    include_robot = False
    include_lane=True
    
    node_type = 'PEDESTRIAN'
    version='mini'
    phase = 'train'
    visualization = False

    if include_lane:
        INTERACTION_RADIUS[('VEHICLE', 'VEHICLE')] = 0
        INTERACTION_RADIUS[('VEHICLE', 'LANE')] = 5
        INTERACTION_RADIUS[('LANE', 'VEHICLE')] = 5
        
    node_type_list = NODE_TYPES
    data_dir = '../../datasets/nuscenes/processed_data'
#%%
    # for phase in ['train', 'val', 'test']:
    processed_sequence, token_map, _ = preprocess_sequence(phase,
                                             data_dir,
                                             node_type_list,
                                             version,
                                             obsv_len=obsv_len,
                                             min_obsv_len=min_obsv_len,
                                             seq_len=seq_len,
                                             min_seq_len=min_seq_len,
                                             skip=skip,
                                             min_ped=min_ped,
                                             include_robot=include_robot,
                                             include_lane=include_lane,
                                             node_type=node_type,
                                             visualization=visualization,
                                             )
    
    # token_matched = [token for _, token in token_map.items() if token in token_list]
    
    #%%
    for i, sequence_dict in enumerate(processed_sequence):
        
        # print(np.unique(sequence['cid']))
        seq_graphs = seq_to_graph(sequence_dict, interaction_radius=INTERACTION_RADIUS)
        obsv_graphs, trgt_graphs = split_graph(seq_graphs)

        gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)

        # make sure there is at least one edges
        if not gt_graphs.number_of_edges()>1:
            continue

        comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].cpu().numpy(), gt_graphs.ndata['tid'].cpu().numpy()))
        # curr_token_list = [token_map[tid] for tid in comm_traj]
        
        # verify 
        # for tid in comm_traj:
        #     obsv_pos = obsv_graphs.ndata['pos'][obsv_graphs.ndata['tid']==tid]
        #     gt_pos = gt_graphs.ndata['pos'][gt_graphs.ndata['tid']==tid]
        #     instance, sample = token_map[tid].split('_')
        #     current_annotation = helper.get_sample_annotation(instance, sample)
        #     history_sequence = helper._iterate(current_annotation, config.seconds, direction='prev')[::-1]
        #     history_sequence = history_sequence[-obsv_len+1:] + [current_annotation]
        #     future_sequence = helper._iterate(current_annotation, config.seconds, direction='next')
        #     node_history = np.array([annotation['translation'][:2] for annotation in history_sequence])
        #     node_futures = np.array([annotation['translation'][:2] for annotation in future_sequence])
        #     assert len(gt_pos)==len(node_futures)
            
        # separate lane node and vehicle nodes
        # obsv_graphs = dgl.node_subgraph(obsv_graphs, obsv_graphs.nodes()[obsv_graphs.ndata['cid']==NODE_TYPES.index(node_type)])
        # lane_graphs = dgl.node_subgraph(trgt_graphs, trgt_graphs.nodes()[trgt_graphs.ndata['cid']==NODE_TYPES.index('LANE')])
        # trgt_graphs = dgl.node_subgraph(trgt_graphs, trgt_graphs.nodes()[trgt_graphs.ndata['cid']==NODE_TYPES.index(node_type)])
        
        
        # plt.close('all')
        # fig, ax = network_draw(dgl.batch([obsv_graphs, gt_graphs]), show_node_label=True, show_edge_label=True, node_label='tid', show_legend=True, font_size=8, node_size=100, fig_name='seq')
        # fig.savefig('../../vis_graphs/preprocessing/{}_scene{}_frame{}.png'.format(phase, seq_graphs.ndata['sid'][0].item(), i), dpi=300, bbox_inches='tight')
        
        # vehicle_graph = dgl.node_subgraph(seq_graphs, seq_graphs.nodes()[seq_graphs.ndata['cid']!=2])
        # network_draw(vehicle_graph, show_node_label=False, node_label='tid', show_legend=True, font_size=8, fig_name='vehicle')
        # # plt.show()
        # lane_graphs = dgl.node_subgraph(seq_graphs, seq_graphs.nodes()[seq_graphs.ndata['cid']==2])
        # network_draw(lane_graphs, show_node_label=False, node_label='tid', show_legend=True, font_size=8, fig_name='lane')
        break
