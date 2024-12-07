#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 07:36:14 2021

@author: loc
"""

# compute mean and std based on processed_data
import os
import dill
import pandas as pd
from collections import defaultdict
import numpy as np
from preprocess_sequence import motion_kinematics
version = 'full'
data_dir = '../../datasets/nuscenes/processed_data'
phase = 'train'

mini_string = ''
if 'mini' in version:
    mini_string = '_mini'
data_dict_path = os.path.join(data_dir, 'nuScenes_' + phase + mini_string + '_full.pkl')

with open(data_dict_path, 'rb') as f:
    scenes_data, _ = dill.load(f)


node_type_states = {}
for state in ['pos', 'vel', 'acc', 'hed']:
    node_type_states[state] = defaultdict(list)

node_type_stats = {}
node_type_stats['mean'] = defaultdict(list)
node_type_stats['std'] = defaultdict(list)

for scene_name, data in scenes_data.items():
    
    
    for node_type in data['type'].unique():
        
        node_type_data = data[data['type']==node_type][['x', 'y']].values
        x_mean, y_mean = node_type_data.mean(0).round(0)
        x_std, y_std = node_type_data.std(0).round(0)
            
        node_type_stats['mean'][node_type].append([x_mean, y_mean])
        node_type_stats['std'][node_type].append([x_std, y_std])
            
        unique_nodes = np.unique(data[data['type']==node_type]['node_id'].values)
        
        for node_id in unique_nodes:
            node_data = data[data['node_id']==node_id][['x', 'y']].values
            node_data = node_data-np.array([x_mean, y_mean])
            if len(node_data)>2:
                p, _, v, a, h = motion_kinematics(node_data, dt=0.5)
                
                node_type_states['pos'][node_type].append(p)
                node_type_states['vel'][node_type].append(v)
                node_type_states['acc'][node_type].append(a)
                node_type_states['hed'][node_type].append(h)
            

        
for node_type in ['PEDESTRIAN', 'VEHICLE']:
    
    print(node_type)
    node_type_mean = np.mean(node_type_stats['mean'][node_type], axis=0).round(0)
    node_type_std = np.mean(node_type_stats['std'][node_type], axis=0).round(0)
    print('mean:', node_type_mean, 'std:', node_type_std)
    
    for state in ['pos', 'vel', 'acc', 'hed']:
        node_type_state = np.concatenate(node_type_states[state][node_type], axis=0)
        node_type_state_mean = node_type_state.mean(0)
        node_type_state_std = node_type_state.std(0)
        print(state, 'mean:', node_type_state_mean, 'std:', node_type_state_std)

# NOTE: the prameters are for whole node lengths that ranges from 0 - 40. Thus this standardization parameters for (pos, rel)
# may be higher than that of actual prediction sequence which is of length 12, whereas vel, acc and head should be the global
# for any sequence length. 

# iirav@iirav:/mnt/dl-asoro/src/nuscenes$ python data_stats.py --test
# Using backend: pytorch
# PEDESTRIAN
# mean: [1130. 1341.] std: [23. 21.]
# pos mean: [0.01227758 0.04773022] std: [27.45380113 26.11627687]
# vel mean: [-0.03790621  0.0983699 ] std: [0.77600581 0.74508554]
# acc mean: [0.00261499 0.00407175] std: [0.2314115  0.27828747]
# hed mean: [-0.01617637  0.0649708 ] std: [0.53296765 0.50985752]

# VEHICLE
# mean: [1182. 1386.] std: [35. 32.]
# pos mean: [0.01142476 0.03342109] std: [37.96032642 35.14679615]
# vel mean: [-0.04727632  0.07179681] std: [4.49514435 4.02348365]
# acc mean: [-0.00509281 -0.00752906] std: [0.75645008 0.67347952]
# hed mean: [-0.02003382  0.01870408] std: [0.57577053 0.53778409]
# iirav@iirav:/mnt/dl-asoro/src/nuscenes$ python data_stats.py  --train
# Using backend: pytorch
# PEDESTRIAN
# mean: [1090. 1297.] std: [23. 20.]
# pos mean: [0.00083313 0.00767852] std: [26.91882698 25.0043608 ]
# vel mean: [-0.09973226  0.02668451] std: [0.83930766 0.74830169]
# acc mean: [-0.00038294 -0.00074935] std: [0.2537066  0.22613068]
# hed mean: [-0.06258206  0.01511344] std: [0.55898284 0.50293753]
# VEHICLE
# mean: [1105. 1296.] std: [34. 29.]
# pos mean: [ 0.00695291 -0.01192071] std: [36.04883091 32.15010249]
# vel mean: [ 0.10509559 -0.11422786] std: [4.69733911 4.04085499]
# acc mean: [-0.00280531 -0.00288383] std: [1.08379564 0.90605546]
# hed mean: [-0.00140959 -0.01261924] std: [0.58952118 0.5325017 ]
