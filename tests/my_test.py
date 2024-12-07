#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:07:31 2021

@author: loc
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (1 - path_distance / path_length), path_length, path_distance

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

all_data = {}
num_peds = defaultdict(int)
ped_ids = []
# for phase in ['test']:
for data_set in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    phase = 'test'
    data_dir = '../datasets/pedestrians/' + data_set + '/' + phase + '/'
    
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    print(all_files)
    
    for file in all_files:
        seq_name = os.path.basename(file)[:-4]
        data = read_file(file, delim='\t')
        
        if data_set in all_data:
            all_data[data_set] = np.concatenate([all_data[data_set], data], axis=0)
        else:
            all_data[data_set] = data
    
        
        #% plot  dataset
        unique_ids = np.unique(data[:, 1])
        print('Num of peds:', len(unique_ids))
        
        frames = data[:, 0]
        for _id in unique_ids:
            ped_frames = data[:, 0][data[:, 1]==_id]
            traj = data[:, 2:4][data[:, 1]==_id]
            curv, path_len, path_distance = trajectory_curvature(traj)
            # print(curv)

            # if not len(traj)>19:
            #     continue
            
            # if not path_distance>0:
            #     continue
            
            # if not (1-path_distance/path_len)>0.01:
            #     continue

        
            # if not curv>0.01:
            #     continue
            
            ped_ids.append(_id)
            num_peds[data_set] += 1
            
            # plt.plot(traj[:, 0], traj[:, 1])
            
            # break
            # sys.exit()
    
        plt.figure(seq_name)
        for pid in unique_ids:
            ped_traj = data[data[:, 1]==pid, :]
            if not ped_traj.shape[0]>8:
                continue
            plt.plot(ped_traj[:, 2], ped_traj[:, 3])
            # break
    
print("Total peds:", sum([v for k, v in num_peds.items()]))
print("Ped ids:", len(np.unique(ped_ids)))

#%%
# data stats in each seq
data_stats = defaultdict(dict)
for k, v in all_data.items():
    pos = v[:, 2:4]
    xs = pos[:, 0]
    ys = pos[:, 1]
    
    data_stats[k]['x_center'] = np.mean(xs).round(3)
    data_stats[k]['y_center'] = np.mean(ys).round(3)

    data_stats[k]['x_scale'] = np.std(xs).round(3)
    data_stats[k]['y_scale'] = np.std(ys).round(3)

    data_stats[k]['x_min'] = np.min(xs).round(3)
    data_stats[k]['y_min'] = np.min(ys).round(3)

    data_stats[k]['x_max'] = np.max(xs).round(3)
    data_stats[k]['y_max'] = np.max(ys).round(3)


concat_data = np.concatenate([v[:, 2:4] for k, v in all_data.items()], 0)

#%% test goal using CV
import warnings
warnings.filterwarnings("ignore")

import math
import random
import numpy as np
import sys
import os
import dgl.function as fn
import matplotlib.pyplot as plt

import torch
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
from data.preprocess_utils import *

from utils.misc import *
from utils.graph_utils import *
from utils.train_utils import *
from utils.config import *

from visualization.vis_graph import network_draw
from visualization.vis_traj import plot_kde
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import interp1d

# if __name__=='__main__':
args = parse_args()

# args.config = './configs/base/pedestrians.py'
args.config = './configs/pedestrian_stdec_sf.py'

cfg = Config.fromfile(args.config)
cfg_dict = cfg.serialize
print(cfg_dict)

torch.cuda.init()
device = torch.device('cuda:0')

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if device.type=='cuda':
    torch.cuda.manual_seed_all(cfg.seed)
#%%
# args.update(cfg_dict)

from data.dataset import PedestriansDataset, NuScenesDataset, WaymoDataset

all_dsets = {'pedestrians':PedestriansDataset, 'nuscenes': NuScenesDataset, 'waymo': WaymoDataset}
cfg.node_types = ['PEDESTRIAN']
cfg.include_lane = False

version = ''
# version = 'mini'
# version = 'full'
# data_dir = '../../datasets/nuscenes/processed_data'
data_dir = '../datasets/pedestrians/eth/'
phase = 'test'
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
                                skip=cfg.skip if phase=='train' else 0,
                                aug_prob=0.5 if phase=='train' else 0,
                                min_agents = cfg.min_agents,
                                # include_robot=cfg.include_robot,
                                include_lane=cfg.include_lane,
                                # pos_enc=cfg.pos_enc,
                                # pos_enc_dim=cfg.pos_enc_dim,
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
obsv_traj = defaultdict(list)
gt_traj = defaultdict(list)
pred_traj = defaultdict(list)
ade, fde = defaultdict(list), defaultdict(list)

SAMPLE_GOAL = 1

for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):
    # remove nodes from trgt_graphs that are not present in obsv graphs
    gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)

    if not gt_graphs.number_of_edges()>1:
        continue
    
    comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].numpy(), gt_graphs.ndata['tid'].numpy()))
    
    # if not len(comm_traj)>5:
    #     continue

    init_pos = torch.stack([pos[-1] for pos in get_ndata(obsv_graphs, 'pos', comm_traj)])
    init_vel = torch.stack([vel[-1] for vel in get_ndata(obsv_graphs, 'vel', comm_traj)])
    
    _, node_idx, time_idx = traj_to_graph_data(gt_graphs, traj_ids=comm_traj)
    
    init_pos = init_pos.unsqueeze(1).repeat(1, cfg.pred_seq_len, 1)[node_idx, time_idx, :]
    init_vel = init_vel.unsqueeze(1).repeat(1, cfg.pred_seq_len, 1)[node_idx, time_idx, :]
    
    pred_pos = init_pos + init_vel*gt_graphs.ndata['ntx'].sub(cfg.obsv_seq_len-1).view(-1, 1)*cfg.dt
    err = torch.linalg.norm(gt_graphs.ndata['pos']-pred_pos, dim=-1, keepdim=True)
    
    # gt_pos = get_ndata(gt_graphs, 'pos', comm_traj)
    # pred_pos = ndata_to_traj(gt_graphs, pred_pos)    

    # pred_pos_cv = [constant_velocity(init_pos[i], init_vel[i], pred_steps=len(gt_pos[i]), time_step=0.4)[:len(gt_pos[i])] for i in range(len(gt_pos))]
        
    ped_err = [np.mean(e) for e in ndata_to_traj(gt_graphs, err)]
    
    ade[iter].append(ped_err)
    
    # cmap = plt.cm.get_cmap(name='Set1', lut=len(comm_traj))
    
    if SAMPLE_GOAL:
        # extract final goal
        goal = torch.stack([pred_pos[gt_graphs.ndata['tid']==tid][-1] for tid in comm_traj])
        
        # print('goals**\n', goals)
        goal_dist = torch.distributions.normal.Normal(goal, scale=torch.tensor([0.8, 1.4]).view(-1, 2))
        for i in range(10):
            sampled_goal = goal_dist.sample()
            # print(sampled_goal)
            sampled_goal = sampled_goal.unsqueeze(1).repeat(1, cfg.pred_seq_len, 1)[..., node_idx, time_idx, :]
            pred_vel1 = (sampled_goal - init_pos)/(gt_graphs.ndata['ftl']*cfg.dt).view(-1, 1)
            pred_pos1 = init_pos + pred_vel1*gt_graphs.ndata['ntx'].sub(cfg.obsv_seq_len).add(1).view(-1, 1)*cfg.dt
            err1 = torch.linalg.norm(gt_graphs.ndata['pos']-pred_pos1, dim=-1, keepdim=True)
            ped_err1 = [np.mean(e) for e in ndata_to_traj(gt_graphs, err1)]
            ade[iter].append(ped_err1)
            
    ade[iter] = np.min(ade[iter], axis=0)# (_, num_peds)
    # break
    
ade_min = [np.mean(v) for k, v in ade.items()]
print('ADE:', np.mean(ade_min))

    
#%%

obsv_traj = defaultdict(list)
gt_traj = defaultdict(list)
pred_traj = defaultdict(list)
ade, fde = defaultdict(list), defaultdict(list)

SAMPLE_GOAL = 1

for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):

    gt_graphs = remove_redundant_nodes(trgt_graphs, obsv_graphs)

    if not gt_graphs.number_of_edges()>1:
        continue
    
    comm_traj = np.sort(np.intersect1d(obsv_graphs.ndata['tid'].numpy(), gt_graphs.ndata['tid'].numpy()))
    
    curr_ade, curr_fde = [], []
    
    cmap = plt.cm.get_cmap(name='Set1', lut=len(comm_traj))
    # fig, axs = plt.subplots()
    for i, tid in enumerate(comm_traj):
        
        obsv_idxs = obsv_graphs.ndata['tid']==tid
        obsv_pos = obsv_graphs.ndata['pos'][obsv_idxs]
        
        gt_idxs = gt_graphs.ndata['tid']==tid        
        gt_pos = gt_graphs.ndata['pos'][gt_idxs]
        
        if not len(obsv_pos)>1:
            continue

        if not len(gt_pos)>1:
            continue
        
        # crv, pl, pd = trajectory_curvature(obsv_pos.numpy())        
        # if pl<1:
        #     continue
        
        init_pos = obsv_pos[-1]
        init_vel = obsv_graphs.ndata['vel'][obsv_idxs][-1]
        init_acc = obsv_graphs.ndata['acc'][obsv_idxs][-1]
        

        # # consider direction from start pos 
        # obsv_len = len(obsv_pos)
        # if obsv_len>=4:
        #     # sys.exit()
        #     stride = 3
        #     diff = obsv_pos[-1] - obsv_pos[-4]
        # else:
        #     stride = obsv_len-1
        #     diff = obsv_pos[-1] - obsv_pos[-obsv_len]
            
        # speed = torch.linalg.norm(diff) / (stride * cfg.dt)
        # theta = np.arctan2(diff[1], diff[0])
        # init_vel = np.array([speed*np.cos(theta), speed*np.sin(theta)])
        
        # constant vel ADE-0.282
        pred_pos = constant_velocity(init_pos, init_vel, init_acc, pred_steps=len(gt_pos), time_step=0.4)[:len(gt_pos)]
        
        # # interpolation ADE- 0.3709
        # time = list(range(len(obsv_pos)))
        # f = interp1d(x=time, y=[obsv_pos[:, 0].tolist(), obsv_pos[:, 1].tolist()], fill_value='extrapolate')
        # pred_pos = torch.from_numpy(np.array([f(t) for t in np.arange(time[-1]+1, time[-1]+1+len(gt_pos), 1)]))
        
        # # kalman ADE - 0.590
        # pred_pos = torch.from_numpy(kalman_predict(obsv_pos.numpy(), pred_steps=len(gt_pos)))
        
        err = torch.linalg.norm(gt_pos - pred_pos, dim=-1)
        ped_ade = [err.mean().item()]
        ped_fde = [err[-1].item()]
        # sys.exit()

        obsv_traj[iter].append(obsv_pos.cpu().numpy())
        pred_traj[iter].append(pred_pos.cpu().numpy())
        gt_traj[iter].append(gt_pos.cpu().numpy())
        
        # color = np.random.random((3,))
        # color = cmap(i)
        # plt.plot(obsv_pos[:, 0], obsv_pos[:, 1], '', color=color)
        # plt.plot(gt_pos[:, 0], gt_pos[:, 1], '-*', color=color)
        # plt.plot(pred_pos[:, 0], pred_pos[:, 1], '--*', markersize=3, color=color)
        
        if SAMPLE_GOAL:
            # scale = F.softmax(1/(obsv_pos.std(0) + 1e-9), dim=-1)
            # scale =1/(obsv_pos.std(0).sigmoid() + 1e-6)
            goal_dist = torch.distributions.normal.Normal(pred_pos[-1], scale=1)
    
            # for goal in goal_list:
            for i in range(10):
                goal = goal_dist.sample()
                dx = (torch.tensor(goal) - init_pos)/len(gt_pos)
                pred = torch.stack([init_pos + (t+1)*dx for t in range(len(gt_pos))])
                err = torch.linalg.norm(gt_pos - pred, dim=-1)
                ped_ade.append(err.mean().item())
                ped_fde.append(err[-1].item())
                # plt.plot(pred_traj[:, 0], pred_traj[:, 1], '--s', markersize=1, color=color)
        # sys.exit()
        
        ade[iter].append(np.min(ped_ade))
        fde[iter].append(np.min(ped_fde))
        
    # plt.show()
    # plt.close('all')
    # sys.exit()
    # break

ade_mean = [np.mean(v) for k, v in ade.items()]
fde_mean = [np.mean(v) for k, v in fde.items()]
print('ADE:', np.mean(ade_mean))    
print('FDE:', np.mean(fde_mean))    
#%%
# frame with max error
max_err_idx = np.argmax(ade_mean)

cmap = plt.cm.get_cmap(name='Set1', lut=10)
i = 0
for obsv_pos, gt_pos, pred_pos in zip(obsv_traj[max_err_idx], gt_traj[max_err_idx], pred_traj[max_err_idx]):
    color = cmap(i)
    plt.plot(obsv_pos[:, 0], obsv_pos[:, 1], '', color=color)
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], '-*', color=color)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], '--*', markersize=3, color=color)
    i+=1
    
# plt.close('all')
#%%
import os
import sys
print(os.path.realpath(__file__))
from utils.misc import copy_src
rel_path = '../../experiments/nuscenes/'
stdec_dir = '../../out/nuscenes/trial_12/run-61_STDec_eh256_dh256_el4_dl4_lr0.0003_nodewt1.0_edgewt0.0_l2_loss_l1_loss_pos_dist_VEHICLE_semi_idist20.0_lane/'
copy_src(rel_path, stdec_dir + '/experiments/')
sys.exit()
#%% load nuscene
import sys
nu_path = '../../nuscenes-devkit/python-sdk/'
if not nu_path in sys.path:
    sys.path.append(nu_path)
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.map_expansion.bitmap import BitMap

from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.config import load_prediction_config, PredictionConfig
from nuscenes.eval.prediction.splits import get_prediction_challenge_split, create_splits_scenes

import numpy as np

VERSION = 'full'

# prepare nusc
if 'mini' in VERSION:
    dataroot = '../../datasets/nuscenes/v1.0-mini'
    split_name = 'mini_val'
    nusc_version = 'v1.0-mini'
else:
    dataroot = '../../datasets/nuscenes/v1.0-trainval_meta'
    split_name = 'val' # split must be one of ['train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track']
    nusc_version = 'v1.0-trainval'
    
print("split:", split_name)
# token_list = get_prediction_challenge_split(split_name, dataroot)

nusc = NuScenes(version=nusc_version, dataroot=dataroot)
helper = PredictHelper(nusc)
config = load_prediction_config(helper, config_name='predict_2020_icra.json')

token_list = get_prediction_challenge_split(split_name, dataroot)
print('Number of token:', len(token_list))


#%%
x = 873
y = 1286
print('Road objects on selected point:', nusc_map.layers_on_point(x, y), '\n')
# print('Next road objects:', nusc_map.get_next_roads(x, y))

nusc_map.render_next_roads(x, y, figsize=1, bitmap=bitmap)
#%%
nusc_ts = []
scene_ids = []
for i, scene in enumerate(nusc.scene):
    scene_ts = []
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    while sample['next']:
        scene_ts.append(int(sample['timestamp']))
        sample = nusc.get('sample', sample['next'])
    nusc_ts.append(scene_ts)
    scene_ids.append(scene['name'].split('-')[-1])
    
#%% plot lane and records


border = 10
patch_box = (curr_seq_data['x'].min()-border, curr_seq_data['y'].min()-border, curr_seq_data['x'].max()+border, curr_seq_data['y'].max()+border)
lane_records_in_patch = nusc_map.get_records_in_patch(patch_box, ['lane', 'lane_connector'], mode='intersect',)
lane_poses = []
for lane_token in lane_records_in_patch['lane']:
    my_lane = nusc_map.arcline_path_3.get(lane_token, [])
    discretized = np.array(discretize_lane(my_lane, resolution_meters=1))
    lane_poses.append(discretized)
    
for lane_token in lane_records_in_patch['lane_connector']:
    my_lane = nusc_map.arcline_path_3.get(lane_token, [])
    discretized = np.array(discretize_lane(my_lane, resolution_meters=1))
    lane_poses.append(discretized)


#%% get closest lanes to nodes
curr_node_data = [curr_seq_data[curr_seq_data['node_id']==node_id] for node_id in curr_nodes ]

# fig = plt.figure()
closest_lane_tokens = []
for node_data in curr_node_data:
    if not len(node_data)>min_seq_len:
        continue
    
    pos = node_data[['x', 'y']].iloc[int(len(node_data)/2)]
    closest_lane = nusc_map.get_closest_lane(pos.x, pos.y, radius=5)
    if len(closest_lane)>0 and closest_lane not in closest_lane_tokens:
        closest_lane_tokens.append(closest_lane)
            
    # first_last_pos = [node_data[['x', 'y']].iloc[0], node_data[['x', 'y']].iloc[-1]]
    # for pos in first_last_pos:
    #     closest_lane = nusc_map.get_closest_lane(pos.x, pos.y, radius=5)
    #     if len(closest_lane)>0 and closest_lane not in closest_lane_tokens:
    #         closest_lane_tokens.append(closest_lane)

    #     continue
    # break

pose_lists = []
for lane_token in closest_lane_tokens:
    my_lane = nusc_map.arcline_path_3.get(lane_token, [])
    discretized = np.array(discretize_lane(my_lane, resolution_meters=1))
    lane_poses.append(discretized)

#%% plot lanes
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
i = 0
for _, pose_list in lane_poses.items():
    if len(pose_list) > 0:
        plt.plot(pose_list[:, 0], pose_list[:, 1], label=i)
    i+=1
# plt.legend()

#%%

for _, node_id in enumerate(curr_nodes):
    
    curr_node_seq = curr_seq_data[curr_seq_data['node_id']==node_id]
    if not curr_node_seq.iloc[0]['type']=='VEHICLE':
        continue
    xs, ys = curr_node_seq['x'].values, curr_node_seq['y'].values
    plt.plot(xs, ys, color='r', linestyle='dashed', marker='*', markersize=1)
    plt.plot(xs[0], ys[0], color='r', marker='o', markersize=4)
    

#%% plot current lane and trajectory
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.close('all')
fig, ax = plt.subplots()

for lane_token, pose_list in curr_lane_poses.items():
    if len(pose_list) > 0:
        plt.plot(pose_list[:, 0], pose_list[:, 1], label=lane_token)


x_min, y_min = np.concatenate(curr_nodes_pos).min(0) - 10
x_max, y_max = np.concatenate(curr_nodes_pos).max(0) + 10
    
rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)


node_color = np.random.random((len(curr_nodes_pos), 3))
for i, pos in enumerate(curr_nodes_pos):
    xs, ys = pos[:, 0], pos[:, 1]
    plt.plot(xs, ys, color=node_color[i], linestyle='dashed', marker='*', markersize=2, label=i)
    plt.plot(xs[0], ys[0], color='r', marker='s', markersize=4)
    
    # pos = gt_futures[i]
    # xs, ys = pos[:, 0], pos[:, 1]
    # plt.plot(xs, ys, color=node_color[i], linestyle='dashed', marker='*', markersize=2)

    for lane in curr_node_lanes[i]:
        pose_list = curr_lane_poses[lane]
        if len(pose_list) > 0:
            plt.plot(pose_list[:, 0], pose_list[:, 1], label=lane_token)

# plt.legend()
plt.show()
# plt.axis('off')

#%% plto lanes and node pos
fig, ax = plt.subplots()
for lane_token, lane_pos in lane_poses.items():
    if len(lane_pos) > 0:
        plt.plot(lane_pos[:, 0], lane_pos[:, 1], label=lane_token, zorder=5)

x_min, y_min = np.concatenate(curr_nodes_pos).min(0) - 10
x_max, y_max = np.concatenate(curr_nodes_pos).max(0) + 10
    
rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

i = 0
node_color = np.random.random((len(curr_nodes_pos), 3))
for pos in curr_nodes_pos:
    xs, ys = pos[:, 0], pos[:, 1]
    plt.plot(xs, ys, color=node_color[i], linestyle='dashed', marker='*', markersize=3, label=i, zorder=2)
    plt.plot(xs[0], ys[0], color=node_color[i], marker='s', markersize=6)
    i+=1
plt.show()
# plt.legend()

#%% trim lanes outside of curr node pos limits

min_values = np.concatenate(curr_nodes_pos).min(0) - 10
max_values = np.concatenate(curr_nodes_pos).max(0) + 10

# max_values = np.array([curr_patch[2], curr_patch[3]])
# min_values = np.array([curr_patch[0], curr_patch[1]])

# lane_poses = {lane:lane_pos[np.logical_and(lane_pos>min_values, lane_pos<max_values).all(-1)] for lane, lane_pos in curr_lane_poses.items() if len(lane_pos)>0}
for lane_token, lane_pos in lane_poses.items():
    valid_idxs = np.logical_and(lane_pos>min_values, lane_pos<max_values).all(-1)
    lane_poses[lane_token] = lane_pos[valid_idxs]

lane_poses = {lane:lane_pos for lane, lane_pos in lane_poses.items() if len(lane_pos)>0}

#%% filter lanes

filtered_lanes = {}
for lane, lane_pos in curr_lane_poses.items():
    for node_pos in curr_nodes_pos:
        d = euclidean_distances(node_pos, lane_pos)
        print(d.min())
        if d.min()<3:
            filtered_lanes[lane] = lane_pos
        
    
fig, ax = plt.subplots()
for lane, lane_pos in filtered_lanes.items():
    if len(lane_pos) > 1:
        plt.plot(lane_pos[:, 0], lane_pos[:, 1])

rect = Rectangle((curr_patch[0], curr_patch[1]), curr_patch[2]-curr_patch[0], curr_patch[3]-curr_patch[1], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

i = 0
node_color = np.random.random((len(curr_nodes_pos), 3))
for pos in curr_nodes_pos:
    xs, ys = pos[:, 0], pos[:, 1]
    plt.plot(xs, ys, color=node_color[i], linestyle='dashed', marker='*', markersize=4, label=i, zorder=2)
    plt.plot(xs[0], ys[0], color=node_color[i], marker='s', markersize=6)
    i+=1
plt.show()
# plt.legend()

#%%
plt.figure()
xs, ys = node_seq[:, 0], node_seq[:, 1]
plt.plot(xs, ys, marker='*', markersize=2, label='obsv_gt_pos')
plt.plot(xs[0], ys[0], color='r', marker='o', markersize=4, label='start')

# xs, ys = node_futures[:, 0], node_futures[:, 1]
# plt.plot(xs, ys, marker='*', markersize=2, label='future_gt_pos')

# xs, ys = node_cv_futures[:, 0], node_cv_futures[:, 1]
# plt.plot(xs, ys, marker='*', markersize=2, label='future_cv_pos')

# xs, ys = lane_points[:, 0], lane_points[:, 1]
# plt.plot(xs, ys, marker='*', markersize=2, label='interp_lane_pos')

# #%%
# xs, ys = node_pos[:, 0], node_pos[:, 1]
# plt.plot(xs, ys, marker='*', linestyle='dashed', markersize=4, label='node_pos')
# plt.plot(xs[0], ys[0], marker='s', markersize=4, label='node_start')

# for lane, lane_pos in lane_poses.items():
for lane in node_lanes:
    lane_pos = curr_lane_poses[lane]
    # plt.plot(lane_pos[:, 0], lane_pos[:, 1], label='node_lane')
    plt.plot(lane_pos[:, 0], lane_pos[:, 1], label=lane[:6])

# plt.legend()



#%% node_pos and lane pos only
plt.plot(node_pos[:, 0], node_pos[:, 1], color='orange', linestyle='dashed', markersize=2, label='node')
plt.plot(node_pos[0, 0], node_pos[0, 1], color='orange', marker='s', markersize=6)

plt.plot(lane_points[:, 0], lane_points[:, 1], label='lane')
plt.legend()

#%% node_lanes and node poses
fig, ax = plt.subplots()

curr_nodes_pos = gt_history

colors = np.random.random((len(curr_nodes_pos), 3))
for i, node_pos in enumerate(curr_nodes_pos):
    
    xs, ys = node_pos[:, 0], node_pos[:, 1]
    plt.plot(xs, ys, marker='*', linestyle='dashed', color=colors[i], markersize=4, label=i)
    plt.plot(xs[0], ys[0], marker='s', color=colors[i], markersize=4,)


for i, node_paths in curr_node_paths.items():

    for path in node_paths:
        path = [l for l in path if l is not None]
        
        if len(path)>0:
            for lane in path:
                lane_pos = curr_lane_poses[lane]
                plt.plot(lane_pos[:, 0], lane_pos[:, 1])
                # plt.plot(lane_pos[:, 0], lane_pos[:, 1], label=lane[:6], color=color)
                
plt.legend()

#%% plot gt history and node lanes
fig, ax = plt.subplots()

colors = np.random.random((len(gt_history), 3))
for i, node_pos in enumerate(gt_history):
    xs, ys = node_pos[:, 0], node_pos[:, 1]
    plt.plot(xs, ys, marker='*', linestyle='dashed', color=colors[i], markersize=4, label=i)
    plt.plot(xs[0], ys[0], marker='s', color=colors[i], markersize=4,)
    
    # plot gt_futures
    xs, ys = gt_futures[i][:, 0], gt_futures[i][:, 1]
    plt.plot(xs, ys, marker='*', linestyle='dashed', color=colors[i], markersize=4, label=i)

    # plot corresponding lane
    for lane in  curr_node_lanes[i]:
        xs, ys = curr_lane_poses[lane][:, 0], curr_lane_poses[lane][:, 1]
        plt.plot(xs, ys, marker='*', linestyle='solid', color=colors[i], markersize=4, label=i)
        
    # for node_path in curr_node_paths[i]:
    #     for lane in node_path:
    #         if lane:
    #             xs, ys = curr_lane_poses[lane][:, 0], curr_lane_poses[lane][:, 1]
    #             plt.plot(xs, ys, marker='*', linestyle='solid', color=colors[i], markersize=4, label=i)            

#%% plot gt pos vs pred pos



#%% interpolate node_pos

import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
# interpolate
# points = curr_data['pos'][5].T
points = node_pos
x, y = zip(*points)

x1 = savgol_filter(x, 5, 3)
y1 = savgol_filter(y, 5, 3)

plt.close('all')
fig, ax = plt.subplots(dpi=100)
plt.plot(x, y, marker='*', color='r', label='original', markersize=4)
plt.plot(x1, y1, marker='o', color='b', label='interpolated')

# plt.legend();


#%%
import numpy as np
# compute mean and std for each state from grph data
def ndata_to_traj(g, ndata):
    return [ndata[g.ndata['tid']==tid].cpu().numpy() for tid in g.ndata['tid'].unique()]

cat_list = [cid[0] for cid in ndata_to_traj(g, g.ndata['cid'])]
    
for cid in np.unique(cat_list):
    print('cid:{}'.format(cid))
    for state in ['rel', 'vel', 'acc', 'hed']:
        traj_state_list = ndata_to_traj(g, g.ndata[state])
        category_state_list = [traj_state_list[i] for i in range(len(traj_state_list)) if cat_list[i]==cid]
        category_state_array = np.concatenate(category_state_list, 0)
    
        print(' {} -  std:{}, mean:{},'.format(state,
                                           np.std(category_state_array, axis=0),
                                           np.mean(category_state_array, axis=0),
                                           ))
        
        
#%% compute mean and std using data loader
from collections import defaultdict
from utils import node_type_list, ndata_to_traj
import dgl

node_type_pos = defaultdict(list)
node_type_rel = defaultdict(list)
node_type_vel = defaultdict(list)
node_type_acc = defaultdict(list)

for iter, (obsv_graphs, trgt_graphs) in enumerate(dataloader):
    # g = dgl.batch([obsv_graphs, trgt_graphs])
    g = trgt_graphs
    if g.number_of_nodes()<1:
        continue
    traj_cid = [cid[0] for cid in ndata_to_traj(g, g.ndata['cid'])]
    traj_pos = ndata_to_traj(g, g.ndata['pos'])
    traj_rel = ndata_to_traj(g, g.ndata['rel'])
    traj_vel = ndata_to_traj(g, g.ndata['vel'])
    traj_acc = ndata_to_traj(g, g.ndata['acc'])
    for i in range(len(traj_cid)):
        cat = node_type_list[int(traj_cid[i])]
        node_type_pos[cat].append(traj_pos[i])
        node_type_rel[cat].append(traj_rel[i])
        node_type_vel[cat].append(traj_vel[i])
        node_type_acc[cat].append(traj_acc[i])
    # break

for cat in node_type_rel.keys():
    if cat=='ROBOT':
        continue
    print(cat)
    cat_pos = np.concatenate(node_type_pos[cat], axis=0)
    cat_rel = np.concatenate(node_type_rel[cat], axis=0)
    cat_vel = np.concatenate(node_type_vel[cat], axis=0)
    cat_acc = np.concatenate(node_type_acc[cat], axis=0)
    print('pos, mean:{}, std:{}'.format(np.mean(cat_pos, axis=0), np.std(cat_pos, axis=0)))
    print('rel, mean:{}, std:{}'.format(np.mean(cat_rel, axis=0), np.std(cat_rel, axis=0)))
    print('vel, mean:{}, std:{}'.format(np.mean(cat_vel, axis=0), np.std(cat_vel, axis=0)))
    print('acc, mean:{}, std:{}'.format(np.mean(cat_acc, axis=0), np.std(cat_acc, axis=0)))
    

# full train set 
# PEDESTRIAN
# pos, mean:[0.03203439 1.25984506], std:[21.34795297 20.91286376]
# rel, mean:[0.04127825 0.25434777], std:[2.24739922 2.20658356]
# vel, mean:[0.0123837  0.11349105], std:[0.77561827 0.75268706]
# acc, mean:[0.00111004 0.00167219], std:[0.17296652 0.16364829]

# VEHICLE
# pos, mean:[1.59390685 0.08311116], std:[33.46850605 31.48649907]
# rel, mean:[ 0.6633998  -0.32587336], std:[12.735499   11.77564974]
# vel, mean:[ 0.19935414 -0.08804276], std:[4.44686655 4.08203327]
# acc, mean:[-0.01859988  0.02621006], std:[0.93394118 0.91669946]

# full test stat
# PEDESTRIAN
# pos, mean:[ 0.84824705 -0.11024577], std:[21.34515617 20.03628902]
# rel, mean:[-0.26151753  0.03908908], std:[2.53991928 2.21329114]
# vel, mean:[-0.10420418  0.01455975], std:[0.83617691 0.7374145 ]
# acc, mean:[0.00140117 0.00023869], std:[0.16065173 0.16015875]

# VEHICLE
# pos, mean:[ 1.47539751 -2.75504508], std:[34.74174407 29.6309413 ]
# rel, mean:[1.07638799 1.07941631], std:[12.38974708 11.04620467]
# vel, mean:[0.43817132 0.41780031], std:[4.24596441 3.81158224]
# acc, mean:[-0.00660413  0.03181569], std:[0.67179083 0.59812365]



    
#%% mean and std based on preprocessed sequence
from collections import defaultdict
from utils import node_type_list

node_type_vel = defaultdict(list)
node_type_acc = defaultdict(list)
node_type_hed = defaultdict(list)
for data_sequence in all_sequence:
    for data_dict in data_sequence:
        
        for idx, cid in enumerate(data_dict['cid']):
            cat = node_type_list[int(cid)]
            node_type_vel[cat].append(data_dict['vel'][idx][:, 0])
            node_type_acc[cat].append(data_dict['acc'][idx][:, 0])
            node_type_hed[cat].append(data_dict['hed'][idx][:, 0])

for cat in node_type_list:
    print('*****', cat, '********')
    print('Velocity    - mean:{}, std:{}'.format(np.mean(node_type_vel[cat], 0), np.std(node_type_vel[cat], 0)))
    print('Acceleration    - mean:{}, std:{}'.format(np.mean(node_type_acc[cat], 0), np.std(node_type_acc[cat], 0)))
    print('Heading    - mean:{}, std:{}'.format(np.mean(node_type_hed[cat], 0), np.std(node_type_hed[cat], 0)))

#%% compute mean and std based on processed_data
import os
import dill
import pandas as pd
from collections import defaultdict
import numpy as np
from preprocess_sequence import motion_kinematics
version = 'full'
data_dir = '../../datasets/nuscenes/processed_data'
phase = 'test'

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

#%% node type ade/fde for given prediction horizon
from collections import defaultdict
from utils import node_type_list
from metrics import compute_ade, compute_fde

prediction_horizon = 12 # steps, number of seconds * 2
node_type_ade = defaultdict(list)
node_type_fde = defaultdict(list)
for iter in range(len(raw_data_dict)):
    
    if not len(raw_data_dict[iter])>0:
        continue
    
    # obsv_pos = raw_data_dict[iter]['obsv']
    gt_pos = raw_data_dict[iter]['trgt']
    predictions_future = raw_data_dict[iter]['pred']
    
    ade_dict = raw_data_dict[iter]['ade_dict']
    fde_dict = raw_data_dict[iter]['fde_dict']
    
    obsv_graphs = raw_data_dict[iter]['obsv_graphs']
    gt_graphs = raw_data_dict[iter]['trgt_graphs']
    
    obsv_t = obsv_graphs.ndata['tid']  
    
    gt_t = gt_graphs.ndata['tid']

    #common traj in obsv and gt graph
    comm_traj = np.sort(np.intersect1d(obsv_t.cpu().numpy(), gt_t.cpu().numpy()))
    
    assert len(comm_traj)==len(ade_dict)
    for i, tid in enumerate(comm_traj):
        node_cid = obsv_graphs.ndata['cid'][obsv_graphs.ndata['tid']==tid]
        node_type = node_type_list[node_cid[0].item()]
        
        if len(gt_pos[i])!=prediction_horizon:
            continue
        
        node_type_ade[node_type].append(np.min(ade_dict[i]))
        node_type_fde[node_type].append(np.min(fde_dict[i]))
        
for node_type in node_type_list:
    if node_type=='ROBOT':
        continue
    print(node_type)
    print('ade :{}'.format(np.mean(node_type_ade[node_type])))
    print('fde :{}'.format(np.mean(node_type_fde[node_type])))

#%% get node lanes
import matplotlib.pyplot as plt
import torch
import random


# assume that each traj has at least one lane
traj_ids = g.ndata['tid'][g.ndata['cid']==1].cpu().unique().numpy()

lane_nodes = {}
lane_poses = {}
# spatial_eids are incoming edges
for i, tid in enumerate(traj_ids):
    # incoming edges
    traj_nodes = g.nodes()[g.ndata['tid']==tid]
    src_nodes, _ = g.in_edges(traj_nodes)
    
    src_lane_nodes = src_nodes[g.ndata['cid'][src_nodes]==3]
    
    # check for multiple lanes
    lane_tids = g.ndata['tid'][src_lane_nodes].unique()
    if len(lane_tids)>1:
        src_lane_nodes = g.nodes()[g.ndata['tid']==random.choice(lane_tids)]
        
    lane_pos = g.ndata['pos'][src_lane_nodes]
    
    lane_nodes[tid] = src_lane_nodes
    lane_poses[tid] = lane_pos
        
lane_poses_array = torch.zeros((len(comm_traj), 12, 2))
for i, tid in enumerate(comm_traj):
    if not tid in lane_poses:
        continue
    lane_pos = lane_poses[tid]
    lane_poses_array[i, :len(lane_pos), :] = lane_pos
    

traj_poses = {tid:g.ndata['pos'][g.ndata['tid']==tid] for tid in traj_ids}

color = np.random.random((len(comm_traj), 3))
# verify
plt.figure()
for i, tid in enumerate(comm_traj):
    if tid not in traj_poses:
        continue
    
    traj = traj_poses[tid].cpu().numpy()
    
    lane = lane_poses_array[i].cpu().numpy()[:len(traj)]
    
    plt.plot(traj[:, 0], traj[:, 1], marker='o', color=color[i], label='traj_{}'.format(i))
    
    plt.plot(lane[:, 0], lane[:, 1], marker='*', color=color[i], label='lane_{}'.format(i))
    
plt.legend()

# network_draw(g, show_node_labels=True, node_label='tid', show_legend=True, font_size=8, fig_name='g')

#%% check edge standardization
e = seq_graphs.edata['dist']
e_std = standardize_edges(seq_graphs, e)
a_sort = e.argsort(0)

plt.figure()
plt.plot(e[a_sort, 0].numpy(), label='e')
plt.plot(e_std[a_sort, 0].numpy(), label='e_std')
plt.legend()

#%%
import matplotlib.pyplot as plt
import numpy as np

max_steps = 100000
def exponential_weight(step, total_steps=100000, decay_steps=10000, initial_value=0.5):
  return initial_value ** ((total_steps - step) / decay_steps)

decay_rate = [exponential_weight(step, max_steps) for step in range(1, max_steps, 1)]
print(decay_rate[0], decay_rate[-1])
plt.plot(np.array(decay_rate))
# plt.plot(np.exp(decay_rate))

#%%
import matplotlib.pyplot as plt
# plot ade per frame
ade_dict_by_frame = {}
for k, raw_data in raw_data_dict.items():
    ade_list = raw_data['ade_list']
    ade_dict_by_frame[k] = ade_list
            
ade_dict_by_frame = [np.mean(val) for k, val in ade_dict_by_frame.items()]
plt.plot(ade_dict_by_frame)

#%%
# select traj based on the max prob
high_prob_vel = []
for i in range(len(comm_traj)):
    p_indices = log_probs[:, i, :].exp().argmax(0)
    vel_p = pred_states['vel'][:, i, :, :][p_indices, np.arange(12), :]
    high_prob_vel.append(vel_p)
high_prob_vel = torch.stack(high_prob_vel, dim=0).unsqueeze(0)

#%%
from nuscenes.map_expansion.arcline_path_utils import _get_lie_algebra
from nuscenes.map_expansion.arcline_path_utils import *

path_length = sum(arcline_path['segment_length'])
radius = arcline_path['radius']

n_points = int(max(math.ceil(path_length / resolution_meters) + 1.5, 2))

resolution_meters = path_length / (n_points - 1)

discretization = []

cumulative_length = [arcline_path['segment_length'][0],
                     arcline_path['segment_length'][0] + arcline_path['segment_length'][1],
                     path_length + resolution_meters]

segment_sign = compute_segment_sign(arcline_path)

poses = _get_lie_algebra(segment_sign, radius)

temp_pose = arcline_path['start_pose']

g_i = 0
g_s = 0.0

for step in range(n_points):

    step_along_path = step * resolution_meters

    if step_along_path > cumulative_length[g_i]:
        temp_pose = pose_at_length(arcline_path, step_along_path)
        g_s = step_along_path
        g_i += 1

    transformation = get_transformation_at_step(poses[g_i], step_along_path - g_s)
    new_pose = apply_affine_transformation(temp_pose, transformation)
    discretization.append(new_pose)

#%%
from data.preprocess_sequence import smooth_points
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
from scipy import stats
import json
import pickle 

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def save_to_csv(val_list, save_path='.'):
    with open(save_path, 'w') as f:
         writer = csv.writer(f)
         writer.writerow(['Step', 'Value'])
         for i, val in enumerate(val_list):
             writer.writerow([i, val])
             
# trial = '../../out/nuscenes/trial_12/'
# run = 'run-60_RGCN_eh256_dh256_el4_dl4_lr0.0003_kldwt1.0_trajwt1.0_nodewt0.0_edgewt0.0_multivariate_normal_mlp_VEHICLE_semi_idist20.0'
# run = 'run-60_DRGCN_eh256_dh256_el4_dl4_lr0.0003_kldwt1.0_trajwt1.0_nodewt0.0_edgewt0.0_multivariate_normal_mlp_VEHICLE_semi_idist20.0'

# trial = '../../out/nuscenes/trial_14/'
# run = 'run-68_RGCN_eh256_dh256_el4_dl4_K10_skip2_kldwt1.0_trajwt1.0_nodewt0.0_edgewt0.0_multivariate_normal_mlp_VEHICLE_full_idist10 .0'
# run = 'run-68_DRGCN_eh256_dh256_el4_dl4_K25_skip2_kldwt1.0_trajwt1.0_nodewt0.0_edgewt0.0_multivariate_normal_mlp_VEHICLE_full_idist10.0'

trial = '../../out/nuscenes/trial_15/'
# run = 'run-74_RGCN_K25_bs64_lr0.0003_skip2_kldwt1.0_trajwt1.0_multivariate_normal_mlp_PEDESTRIAN_full_idist5.0'
run = 'run-74_DRGCN_K25_bs16_lr0.0003_skip2_kldwt1.0_trajwt1.0_multivariate_normal_mlp_PEDESTRIAN_full_idist5.0'
# run = 'run-74_DRGCN_K25_bs64_lr0.0003_skip2_kldwt1.0_trajwt1.0_multivariate_normal_mlp_PEDESTRIAN_full_idist5.0'

out_path = '/home/dl-asoro/Dropbox/PhD Research/ICRA_2023/figures/'

with open (trial + run + '/history.json', 'rb') as f:
    hist = json.load(f)

with open (trial + run + '/args.pkl', 'rb') as f:
    args = pickle.load(f)

# for m in ['train_kld_loss']:
for m in ['train_kld_loss', 'train_traj_loss', 'val_traj_loss']:
    val = hist[m]
    
    val = [v for v in val if v<val[0]]
    
    if args.model=='RGCN' and m=='train_kld_loss':
        val = val[100:]
        val = [v/2 for v in val]
    
    smooth_val = smooth(val, weight=0.96)

    plt.figure()
    plt.title(m)
    plt.plot(val)
    plt.plot(smooth_val)
    # plt.plot(intp_val)
    
    # save_to_csv(smooth_val, out_path + '{}_smooth_{}.csv'.format(args.model, m))

#%% plot Normal distributions

from scipy.stats import norm
from scipy.stats import laplace

np.random.seed(2)

obsv_graphs, trgt_graphs = next(dataloader.__iter__())
seq_graphs = dgl.batch([obsv_graphs, trgt_graphs])
x = seq_graphs.edata['dist'][seq_graphs.edata['spatial_mask']==1].view(-1).numpy()
# x = seq_graphs.ndata['vnorm'].view(-1).numpy()


mu = np.mean(x)
sigma = np.std(x)


s = np.random.normal(mu, sigma, 10000)
count, bins, ignored = plt.hist(s, 30, density=True, alpha=0.5)

gaussian_pdf = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))
plt.plot(bins,  gaussian_pdf, linewidth=2, color='r', label='Normal (loc:{:.1f}, scale:{:.1f})'.format(mu, sigma))


# # laplace 
# s = np.random.laplace(mu, sigma, 1000)

# count, bins = np.histogram(s, 100, density=True)
# laplace_pdf = np.exp(-abs(bins-mu)/sigma)/(2.*sigma)

# plt.plot(bins, laplace_pdf, linewidth=2, color='g', label='Laplace(loc:{:.1f}, scale:{:.1f})')

ax = plt.gca()
# max_x = 1*x.max()
# ax.set_xlim([-max_x, max_x])
ax.axes.yaxis.set_ticks([])
# plt.legend()

#%% visualize p_ij/|p_ij| and v_i/|v_i|
import matplotlib.pyplot as plt
import dgl.function as fn
from visualization.vis_graph import network_draw
g = gt_graphs
nids = [9, 41, 45]
nodes = [g.nodes()[g.ndata['nid']==nid] for nid in nids]
nodes = [node[g.ndata['ntx'][node]>16] for node in nodes]
g = dgl.node_subgraph(gt_graphs, torch.cat(nodes))

nodes = [g.nodes()[g.ndata['ntx']>7]]

g.ndata['dir'] = g.ndata['hed']
# g = gy
g.edata['dir'] = g.edata['diff']/g.edata['dist']
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='dir', fig_name='dir')

g.edata['sigma'] = g.edata['dist'].mul(-1).exp()
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='sigma', fig_name='sigma')

g.apply_edges(fn.u_dot_v('dir', 'dir', 'uv_dir'))
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='uv_dir', fig_name='uv_dir')

g.apply_edges(fn.e_dot_v('dir', 'dir', 'ev_dir'))
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='ev_dir', fig_name='ev_dir', figsize=(5, 5))

g.apply_edges(fn.e_dot_u('dir', 'dir', 'eu_dir'))
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='eu_dir', fig_name='eu_dir', figsize=(5, 5))

g.edata['scaled_ev_dir'] = 0.5*(1-g.edata['ev_dir'])
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_ev_dir', fig_name='scaled_ev_dir', figsize=(5, 5))

g.edata['scaled_eu_dir'] = 0.5*(1-g.edata['eu_dir'])
# network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_eu_dir', fig_name='scaled_eu_dir', figsize=(5, 5))
#%%
fig, axes = plt.subplots(2, 2)

# g.edata['a_ev'] = g.edata['uv_dir'] * g.edata['ev_dir']
g.edata['a_ev'] = g.edata['uv_dir'] + g.edata['ev_dir']
# g.edata['a_ev'] = g.edata['uv_dir'] * torch.sigmoid(g.edata['ev_dir'])
network_draw(g, node_label='_N', show_edge_label=True, edge_label='a_ev', fig_name='a_ev', ax=axes[0, 0])

# g.edata['a_eu'] = g.edata['uv_dir'] * g.edata['eu_dir']
# g.edata['a_eu'] = g.edata['uv_dir'] * torch.sigmoid(g.edata['eu_dir'])
g.edata['a_eu'] = g.edata['uv_dir'] + g.edata['eu_dir']
network_draw(g, node_label='_N', show_edge_label=True, edge_label='a_eu', fig_name='a_eu', ax=axes[0, 1])

g.edata['scaled_a_ev'] = g.edata['uv_dir'] * g.edata['scaled_ev_dir']

network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_a_ev', fig_name='scaled_a_ev', ax=axes[1, 0])

g.edata['scaled_a_eu'] = g.edata['uv_dir'] * g.edata['scaled_eu_dir']
network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_a_eu', fig_name='scaled_a_eu', ax=axes[1, 1])

#%%
fig, axes = plt.subplots(2, 2)

g.edata['a_ev_sigma'] = g.edata['a_ev'] * g.edata['sigma']
# g.edata['a_ev_sigma'] = g.edata['a_ev'] + (g.edata['sigma'] * g.edata['dir'])
# g.edata['a_ev_sigma'] = torch.tanh(g.edata['a_ev']) * g.edata['sigma']
network_draw(g, node_label='_N', show_edge_label=True, edge_label='a_ev_sigma', fig_name='a_ev_sigma', ax=axes[0, 0])

g.edata['a_eu_sigma'] = g.edata['a_eu'] * g.edata['sigma']
# g.edata['a_eu_sigma'] = g.edata['a_eu'] + (g.edata['sigma'] * g.edata['dir'])
# g.edata['a_eu_sigma'] = torch.tanh(g.edata['a_eu']) * g.edata['sigma']
network_draw(g, node_label='_N', show_edge_label=True, edge_label='a_eu_sigma', fig_name='a_eu_sigma', ax=axes[0, 1])

g.edata['scaled_a_ev_sigma'] = g.edata['scaled_a_ev'] * g.edata['sigma']
# g.edata['scaled_a_ev_sigma'] = g.edata['scaled_a_ev'] + (g.edata['sigma'] * g.edata['dir'])
network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_a_ev_sigma', fig_name='scaled_a_ev_sigma', ax=axes[1, 0])

g.edata['scaled_a_eu_sigma'] = g.edata['scaled_a_eu'] * g.edata['sigma']
# g.edata['scaled_a_eu_sigma'] = g.edata['scaled_a_eu'] + (g.edata['sigma'] * g.edata['dir'])
network_draw(g, node_label='_N', show_edge_label=True, edge_label='scaled_a_eu_sigma', fig_name='scaled_a_eu_sigma', ax=axes[1, 1])


#%%
#%%
sampled_indices = np.arange(0, 91, 5)
keep_tracks = sample['state/tracks_to_predict'].numpy()!=-1

state_dict = {}
keep_tracks = sample['state/tracks_to_predict'].numpy()!=-1
sampled_indices = np.arange(0, 91, 5) # new sampling rate will be 2Hz i.e. 0.5 secs
for s in ['id', 'type', 'is_sdc', 'tracks_to_predict']:
    state_dict[s] = sample[f'state/{s}'].numpy()[keep_tracks]

for s in ['x', 'y', 'z', 'velocity_x', 'velocity_y', 'vel_yaw', 'height', 'length', 'width', 'bbox_yaw','timestamp_micros', 'valid'] :
    state_dict[s] = np.concatenate([sample[f'state/past/{s}'].numpy(), #(128, 10))
                                sample[f'state/current/{s}'].numpy(),  # (128, 1)
                                sample[f'state/future/{s}'].numpy(),  # (128, 80)
                                ], -1)[keep_tracks]
    
pos = np.stack([state_dict['x'], state_dict['y']], axis=2)
vel = np.stack([state_dict['velocity_x'], state_dict['velocity_y']], axis=2)
valid = state_dict['valid']
ts = state_dict['timestamp_micros']

pos_sampled = pos[:, sampled_indices, :]
# vel_sampled = vel[:, sampled_indices, :]
# valid_sampled = valid[:, sampled_indices]
ts_sampled = ts[:, sampled_indices]

valid_sampled = np.add.reduceat(valid, sampled_indices, axis=1) # time steps
valid_sampled_rpt = np.stack([valid_sampled, valid_sampled], 2)

vel_sampled = np.add.reduceat(vel * valid[:, :, None], sampled_indices, axis=1) * valid[:, sampled_indices][:, :, None]
vel_sampled = np.divide(vel_sampled, valid_sampled_rpt, where=valid_sampled_rpt>0) 
vel_sampled[valid_sampled_rpt==0] = -1.0


#%%
pos_err = []
pos_err_sampled = []

def compute_pos_from_vel(start_pos, vel, dt):
    rel_pos = np.cumsum(vel, 0)[:-1, :] * dt[:, None]
    pos = np.concatenate([start_pos,  start_pos+rel_pos], axis=0)
    return pos
    
for i in range(len(pos)):
    valid_idx = valid[i]==1
    p = pos[i][valid_idx]
    v = vel[i][valid_idx]
    
    if not len(p)>1:
        continue
    
    t = ts[i][valid_idx]
    dt = (t[1:] - t[:-1]) * 1e-6
    pos_from_vel = compute_pos_from_vel(p[0:1], v, dt)
    
    err = np.linalg.norm(p - pos_from_vel, axis=-1).mean()
    pos_err.append(err)
    
    valid_idx = valid_sampled[i]==1
    sampled_p = pos_sampled[i][valid_idx]
    if not len(sampled_p)>0:
        continue
    sampled_v = vel_sampled[i][valid_idx]
    sampled_t = ts_sampled[i][valid_idx]
    sampled_dt = (sampled_t[1:] - sampled_t[:-1]) * 1e-6

    sampled_p_from_vel = compute_pos_from_vel(sampled_p[0:1], sampled_v, sampled_dt)
    err = np.linalg.norm(sampled_p - sampled_p_from_vel, axis=-1).mean()
    pos_err_sampled.append(err)
        
    # break
print('Mean pos err:', np.mean(pos_err))
print('Mean sampled pos err:', np.mean(pos_err_sampled))

#%% eval interval
import pandas
import matplotlib.pyplot as plt
df = pandas.read_csv('/home/dl-asoro/Downloads/val_ade1.csv')

h = []
last_eval_epoch = 0
last_eval_loss = 0

for e in range(1, df.Step.values[-1] + 1):
    
    eval_condition1 = epoch==1
    eval_condition2 = epoch<=30 and epoch%5==0
    eval_condition3 = epoch>30 and h[-1]<min(h[-5:-1])
    
    if eval_condition1 or eval_condition2 or eval_condition3:
        h.append(val_loss)

        last_eval_epoch = epoch
        
#%%
import pandas
import json

df = pandas.read_csv('/home/dl-asoro/Downloads/val_ade1.csv')
hist_path = '/home/dl-asoro/Desktop/Recon_GCN/out/pedestrians/trial_4/run-19_STDec_GatedGCNLayerSF_h128_l2_K10_K_eval20_bs1_ntx_vel_vnorm_dir_hed_speed_dl-asoro/eth'

with open(hist_path + '/history.json', 'r') as f:
    h = json.load(f)

key= 'test_ade'

eval_epochs = []
eval_ade = []
eval_counter = 0

epochs = []
epoch_ade = {}

test_interval = 5
for epoch in h['epoch']:
    
    # if e==1:
    #     continue
    
    # simulate like in training
    eval_condition1 = epoch==1
    eval_condition2 = epoch<=30 and epoch%test_interval==0#
    eval_condition3 = epoch>30 and h['val_loss'][:epoch][-1]<=min(h['val_loss'][:epoch][-test_interval:-1])# if minimum in last 5 epochs
    
    if eval_condition1 or eval_condition2 or eval_condition3:
        last_eval_epoch = epoch
        eval_ade.append(h['test_ade'][eval_counter])
        eval_epochs.append(epoch)
        eval_counter += 1
    
    # if epoch==85:
    #     break
    
    if epoch%test_interval==0 or epoch==1:
        epochs.append(epoch)
        epoch_ade[epoch] = np.min(eval_ade[min(last_eval_epoch, epoch-1)//test_interval:] )

# assert np.all(np.array(epoch_ade) == np.array(df.Value.values)), 'ade are not equal'
# plt.plot(df.Step.values[1:], df.Value.values[1:], 'r--s', )
# plt.plot(epochs, epoch_val_ade, '-o')
# plt.grid()


    #%%  create full st graph   
    import numpy as np
    
    idist = 5
    nodes_data = {}
    peds_len = [len(pos[:4]) for pos in curr_sequence['pos']]
    for key, val in curr_sequence.items():
        if key in NODE_INFO:
            nodes_data[key] = np.concatenate([[v]*peds_len[i] for i, v in enumerate(val)])
        else:
            nodes_data[key] = np.concatenate([v[:4] for v in val] , 0)
        # except ValueError:
        #     nodes_data[key] = np.array(val)
    
    num_agents = len(curr_sequence['pos'])
    
    nodes_pos = nodes_data['pos']
    nodes_vel = nodes_data['vel']
    nodes_dir = nodes_data['dir']
    
    num_nodes = len(nodes_pos)
    print(f"{num_nodes=}")
    
    # adjacency_matrix
    peds_tid = nodes_data['tid']
    peds_ntx = nodes_data['ntx'].flatten()
    
    # temporal and spatial adjoint matrix 1-edge, 0-no edge
    temporal_matrix = np.logical_and(peds_ntx[None, :]==peds_ntx[:, None]+1, peds_tid[None, :]==peds_tid[:, None])
    spatial_matrix = np.logical_and(peds_tid[None, :]!=peds_tid[:, None], peds_ntx[None, :]==peds_ntx[:, None])
    
    # consider euclidean distance
    diff = nodes_pos[None, :, :] - nodes_pos[:, None, :]  # dst - src, row indices-> src, col_indices->dst, 
    # diff = nodes_pos[:, None, :] - nodes_pos[None, :, :]  #src - dst, row indices-> src, col_indices->dst, note the new axis!
    # edir = diff/(dist + 1e-9)
    edir = unit_vector(diff)
    
    # consider field of view
    full_fov = field_of_view(nodes_dir, edir, twophi=360) # 0-1
    wider_fov = field_of_view(nodes_dir, edir, twophi=200) # 0-1
    narrow_fov = field_of_view(nodes_dir, edir, twophi=50)
    
    # consider for opposite direction
    n_dir = np.dot(nodes_dir, nodes_dir.T)
    
    # adjust fov 
    fov_same_dir = (n_dir>0).astype(int) * wider_fov
    fov_opp_dir = (n_dir<0).astype(int) * narrow_fov 
    
    # interaction between different neighbors
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)
    idist = np.zeros_like(dist)
    for i, cid1 in enumerate(nodes_data['cid']):
        for j, cid2 in enumerate(nodes_data['cid']):
            idist[i, j] = get_interaction_radius(cid1, cid2, INTERACTION_RADIUS)
            
    spatial_matrix = (dist<idist).astype(int)*spatial_matrix + fov_opp_dir*(dist<idist).astype(int)
    
    adj_matrix = np.logical_or(spatial_matrix, temporal_matrix)
    print(f"{np.sum(adj_matrix)=}")
    
    src_ids, dst_ids = np.where(adj_matrix>0)
    g = dgl.graph((src_ids, dst_ids))    
    
    print(f'Nodes:{g.number_of_nodes()}, Edges:{g.number_of_edges()}')
    # Add  features
    for attr in NODE_INFO:
        g.ndata[attr] = torch.tensor(nodes_data[attr], dtype=torch.int64)
        
    for attr in NODE_STATES:
        g.ndata[attr] = torch.DoubleTensor(np.stack(nodes_data[attr], axis=0))
    
    edges_data = {}
    edges_data['diff'] = nodes_data['pos'][dst_ids] - nodes_data['pos'][src_ids]
    edges_data['dist'] = np.linalg.norm(edges_data['diff'], axis=-1, keepdims=True)
    edges_data['dir'] = np.where(edges_data['dist']>0, edges_data['diff']/edges_data['dist'], edges_data['diff'])
    edges_data['temporal_mask'] = temporal_matrix[src_ids, dst_ids]
    edges_data['spatial_mask'] = 1 - edges_data['temporal_mask']
    
    for attr in EDGE_ATTRS:
        g.edata[attr] = torch.DoubleTensor(np.reshape(edges_data[attr], (len(edges_data[attr]), -1)))
    
    # g = g.subgraph(torch.cat([g.nodes()[g.ndata['tid']==tid] for tid in [5, 8]]))
    network_draw(g, node_label='_N', figsize=(8, 8), node_size=500, show_legend=False, alpha=0.9)
        