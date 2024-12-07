#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:18:54 2020

@author: niraj
"""
import math
import os
import sys
import numpy as np
from collections import defaultdict

from data.states import *
from data.preprocess_utils import *

ALL_DSETS = ['students001', 'students003', 'crowds_zara01', 'crowds_zara02', 
             'crowds_zara03', 'biwi_eth', 'biwi_hotel', 'uni_examples']

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def preprocess_sequence(data_dir, phase, obsv_len=8, pred_len=12, min_obsv_len=2, min_seq_len=2, 
                        skip=0, min_peds=1, dt=0.4, padding=False, single_sequence=False):

    print('Preprocessing {} sequences from {}'.format(phase, data_dir))
    #%%
    all_files = os.listdir(os.path.join(data_dir, phase))
    all_files = [os.path.join(data_dir, phase, file_path) for file_path in all_files if '.txt' in file_path]
    # print(all_files)
    
    processed_sequence = []
    
    total_peds = 0
    
    global_tid = 0
    seq_len = obsv_len + pred_len
    # pbar = tqdm(total=len(all_files), position=0) 
    for path in all_files:
        # pbar.update(1)
        scene = os.path.basename(path).split('.')[0]
        
        print('Processing', scene)
        
        data = read_file(path)

        x_mean, y_mean = data[:, 2].mean(), data[:, 3].mean()
        
        #NOTE: Different pedestrian at different scene can have same ID. Make them unique across all scene.
        #first map the scene id to a new global id
        unique_ids = np.unique(data[:, 1])
        unique_ids_new = total_peds + np.arange(len(unique_ids))
        id_map = {pid:pid_new for pid, pid_new in zip(unique_ids, unique_ids_new)}
        #IMPORTANT! Count all unique pedes in each scene
        total_peds += len(unique_ids)
        
        #update id for each row in the scene
        data[:, 1] = np.array([id_map[pid] for pid in data[:, 1]])
        
        frames = np.unique(data[:, 0]).tolist()
        print(f'Total frames: {len(frames)}')

        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])

        # num_sequences = int(math.ceil((len(frames) - seq_len) / skip))
        num_frames = len(frames)
        
        for idx in range(min_obsv_len, num_frames, skip+1):
            
            # print(f'{idx}/{num_frames}')

            start_idx = max(0, idx-obsv_len+1)
            end_idx = min(num_frames, idx+pred_len+1)   
            
            curr_seq = np.concatenate(frame_data[start_idx:end_idx], axis=0)

            curr_peds = np.unique(frame_data[idx][:, 1])
                        
            curr_sequence = defaultdict(list)
            
            num_peds_considered = 0
            for _, pid in enumerate(curr_peds):
                
                ped_seq = curr_seq[curr_seq[:, 1]==pid, :]
                # ped_seq = np.around(ped_seq, decimals=3)
                                
                # index of current frame_num in ped seq
                ped_idx = list(ped_seq[:, 0]).index(frames[idx]) 
                
                ped_history_len = ped_idx  + 1 # node history include current index as well
                ped_future_len = len(ped_seq) - ped_idx  - 1 
                
                assert ped_history_len + ped_future_len == len(ped_seq)
                
                # pad_front and pad front are index value
                pad_front = obsv_len - ped_history_len
                pad_end = obsv_len + ped_future_len
                
                # at least min_obsv_len, pad_front can range from 0 - 19
                if not obsv_len-pad_front>min_obsv_len-1:
                    continue
                
                # at least two gt future pos, avoid error during graph split
                if not pad_end-obsv_len>1:
                    continue
                
                # at least sequence length of min_seq_len
                if not pad_end-pad_front>min_seq_len-1:
                    continue
                
                crv, pl, pd = trajectory_curvature(ped_seq[:, 2:4])
                
                if pd<0.5:
                    ped_seq[:, 2:4] = ped_seq[0, 2:4]

                # if phase=='train':
                #     if not pl>0:
                #         continue

                #     if not (1-pd/pl)>0.01:
                #         continue

                ped_states = node_sequence(ped_seq[:, 2:4], 
                                            dt, 
                                            pad_front=pad_front if padding else 0, 
                                            pad_end=pad_end if padding else len(ped_seq), 
                                            seq_len=seq_len if padding else len(ped_seq), 
                                            obsv_idx=ped_idx,
                                            frames=ped_seq[:, 0],
                                            yaw=None)

                for state in ped_states.keys():
                    curr_sequence[state].append(ped_states[state])
                
                curr_sequence['nid'].append(pid)
                curr_sequence['tid'].append(global_tid)
                curr_sequence['cid'].append(NODE_TYPES.index('PEDESTRIAN'))
                curr_sequence['sid'].append(ALL_DSETS.index(scene.replace('_train', '').replace('_val', '')))
                curr_sequence['htl'].append(ped_history_len)
                curr_sequence['ftl'].append(ped_future_len)
                curr_sequence['dt'].append(dt)
                
                # ntx is computed by seq_to_st_graph if padding else need to compute for seq_to_full_graph
                if not padding:
                    curr_sequence['ntx'].append(np.arange(seq_len)[pad_front:pad_end][:, None])
                
                global_tid+=1
                num_peds_considered+=1
                
                # sys.exit()

            if num_peds_considered>min_peds:
                # for key, _ in curr_sequence.items():
                #     curr_sequence[key] = np.stack(curr_sequence[key] , 0)                

                # standardize pos 
                # curr_sequence['pos'] = curr_sequence['pos'] - np.array([x_mean, y_mean])[None, None, :]
                processed_sequence.append(curr_sequence)
                
                if single_sequence:
                    break
        
        if single_sequence:
            break
    
    # pbar.close()
    # print('\nFinished preprocessing {} sequences in {:.1f}s'.format(self.phase, time.time()-start_time))
    print('Total number of peds:', total_peds)
    print('Total number of sequence:', len(processed_sequence))

    #%%
    return processed_sequence


#%%
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    dset_name = 'eth'
    phase = 'test'
    data_dir = '../datasets/pedestrians/' + dset_name + '/'
    # data_dir = '../datasets/pedestrians/raw/'
    obsv_len=8
    pred_len=12
    seq_len = obsv_len + pred_len
    min_obsv_len = 2
    min_seq_len = 2
    skip=1
    min_peds = 3
    dt = 0.4
    padding = False
    single_sequence = True
    fov_graph = True
    processed_sequence = preprocess_sequence(data_dir, 
                                             phase, 
                                             min_seq_len=min_seq_len,
                                             skip=skip, 
                                             min_peds=min_peds, 
                                             padding=not fov_graph, 
                                             single_sequence = single_sequence
                                             )
    #%%
    from data.preprocess_utils import *
    from visualization.vis_graph import network_draw
    from utils.graph_utils import *
    from data.states import INTERACTION_RADIUS

    
    INTERACTION_RADIUS['PEDESTRIAN', 'PEDESTRIAN'] = 5
    

    for i, sequence_dict in enumerate(processed_sequence):
        
        # if i!=10:
        #     continue
        
        if fov_graph:
            g = seq_to_fov_graph(sequence_dict, INTERACTION_RADIUS)
        else:
            g = seq_to_st_graph(sequence_dict, INTERACTION_RADIUS, seq_len=20, twophi=300)
        
        # print(f"{g.number_of_nodes()=}, {g.number_of_edges()=}")
        # g1, g2 = split_graph(g)
        
        # subnodes = [10, 11, 12, 14, 15]
        # g = g.subgraph(torch.cat([g.nodes()[g.ndata['nid']==nid] for nid in subnodes]))
        
        # g.edata['e_dir'] = -g.edata['dir']
        # g.apply_edges(dgl.function.u_dot_e('dir', 'dir', 've_dir'))
        # g.apply_edges(dgl.function.u_dot_v('dir', 'dir', 'uv_dir'))
        # network_draw(g, node_label='_N', show_edge_label=False, edge_label='dist', fig_name=i, show_legend=False)
        # network_draw(g, node_label='ntx', figsize=(8, 8), node_size=200, show_legend=False, alpha=0.9)
        
        break
        # plt.waitforbuttonpress(-1)
        
