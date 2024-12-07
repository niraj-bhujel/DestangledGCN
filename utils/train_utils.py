import os
import sys
import random
import torch
import pickle
import shutil
import json

import os.path as osp
import numpy as np
from datetime import datetime
from collections import defaultdict

from data.states import NODE_TYPES, STATE_DIMS
from utils.misc import copy_src, create_new_dir
from model.losses import loss_func_param

def set_random_seed(seed):
    #setup seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def logits_to_dict(logits, state_dims: dict):
    states = [s for s, _ in state_dims.items()]
    start_idx = [0] + np.cumsum([state_dims[s] for s in states]).tolist()
    start = {s:start_idx[i] for i, s in enumerate(states)}
    end = {s:start_idx[i]+state_dims[s] for i, s in enumerate(states)}
    return {s:logits[..., start[s]:end[s]] for s in states}

def prepare_edata(g, edge_states):
    edata = {}
    for s in edge_states:
        edata[s] = g.edata[s]
    return edata

def prepare_ndata(g, node_states, standardize_nodes=False, standardization_param=None):
    ndata = {}
    for s in node_states:
        if s in ['ntx', 'nid', 'fid', 'sid', 'cid', 'tid']:
            ndata[s] = g.ndata[s].view(-1, 1)
        else:
            ndata[s] = g.ndata[s]    
    return ndata

def prepare_inputs(g, node_states, edge_states, standardize_nodes=False, standardization_param=None):
    
    x = prepare_ndata(g, node_states, standardize_nodes=False, standardization_param=None)

    if standardize_nodes:
        x = standardize_nodes(g, x, standardization_param, )

    e = prepare_edata(g, edge_states)

    return g, x, e

def get_inputs_dims(input_states):
    return {s:STATE_DIMS[s] for s in input_states}

def get_outputs_dims(output_states, loss_func):
    return {s:loss_func_param(loss_func, STATE_DIMS[s]) for s in output_states}

def get_standaridzation_params(states, standardization):
    
    node_types = list(standardization.keys())

    std_param = defaultdict(dict)
    for node_type in node_types:
        mean = {}
        std = {}
        for state in states:
            for dim in standardization[node_type][state].keys():
                mean[state] = standardization[node_type][state][dim]['mean']
                std[state] = standardization[node_type][state][dim]['std']
                
        std_param[node_type]['mean'] = mean
        std_param[node_type]['std'] = std
        
    return std_param

def standardize_nodes(g, x, standardization_param=None, standardization=None, states=None):
    '''
    x: nodes data to standardiz, (num_nodes, dim)
    cid: category id of the node data ()
    node_types: list of node types
    states: list of states of the node data
    '''
    
    if standardization_param is None:
        if states is None or standardization is None:
            raise Exception('states and standardization should be provided when standardization_param is None')
        standardization_param = get_standaridzation_params(states, standardization)
    
    mean = torch.zeros(x.shape)
    std = torch.ones(x.shape)
    
    for cid in g.ndata['cid'].unique():
        node_type = NODE_TYPES[cid.item()]
        node_mean = standardization_param[node_type]['mean']
        node_std = standardization_param[node_type]['std']
        
        mean[..., g.ndata['cid']==cid, :] = torch.as_tensor(node_mean)#torch.DoubleTensor(cat_mean).to(x.device)
        std[..., g.ndata['cid']==cid, :] = torch.as_tensor(node_std)#torch.DoubleTensor(cat_std).to(x.device)
    
    mean = mean.type_as(x)
    std = std.type_as(x)
    
    return (x-mean)/std

    
def unstandardize_nodes(g, x, standardization_param=None, standardization=None, states=None):
    '''
    x: nodes data to standardiz, (num_nodes, dim) or (K, num_nodes, dim)
    cid: category id of the node data ()
    node_types: list of node types
    states: list of states of the node data
    '''
    
    if standardization_param is None:
        if states is None or standardization is None:
            raise Exception('states and standardization should be provided when standardization_param is None')
        standardization_param = get_standaridzation_params(states, standardization)
    
    mean = torch.zeros(x.shape)
    std = torch.ones(x.shape)
    
    for cid in g.ndata['cid'].unique():
        node_type = NODE_TYPES[cid.item()]
        cat_mean = standardization_param[node_type]['mean']
        cat_std = standardization_param[node_type]['std']
        
        mean[..., g.ndata['cid']==cid, :] = torch.as_tensor(cat_mean)#torch.DoubleTensor(cat_mean).to(x.device)
        std[..., g.ndata['cid']==cid, :] = torch.as_tensor(cat_std)#torch.DoubleTensor(cat_std).to(x.device)

    mean = mean.type_as(x)
    std = std.type_as(x)
    
    return x*std + mean

def standardize_edges(g, e, personal_rad=1.0):
    src, dst = g.edges()
    personal_rad = [INTERACTION_RADIUS[NODE_TYPES[int(c1)], NODE_TYPES[int(c2)]] for c1, c2 in zip(g.ndata['cid'][src], g.ndata['cid'][dst])]
    personal_rad = torch.DoubleTensor(personal_rad).unsqueeze(-1).to(e.device)
    # return e/personal_rad
    return torch.exp(-e**2/(2*personal_rad**2)+1e-9)
    
def unstandardize_edges(e, personal_rad=0.4):
    e[e<0]=1e-6
    return torch.log(e).mul(-2*personal_rad**2)

def preprare_and_backup(args, cfg):
    
    if args.debug:
        return args
    
    if os.path.exists(args.out_path):
        if not args.resume_training:
            if args.overwrite:
                print('Run exists! Overwritting existing run...')
            else:
                raise Exception('Run exists! Continuing to next run...')
                return None            
    else:
        create_new_dir(args.out_path)

    args.summary_dir = create_new_dir(args.out_path + '/summary/', clean=(not args.resume_training))
    
    if args.plot_traj:
        args.plot_traj_dir = create_new_dir(args.out_path + '/plot_traj/training/', clean=(not args.resume_training))

    if args.vis_graph:
        args.vis_graph_dir = create_new_dir(args.out_path + '/vis_graph/training/', clean=(not args.resume_training))

    with open(args.out_path + '/args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    with open(args.out_path + '/cfg.pkl', 'wb') as fp:
        pickle.dump(cfg, fp)

    with open(args.out_path + '/args.json', 'w') as f: 
        json.dump(vars(args), f)

    with open(args.out_path + '/cfg.json', 'w') as fp:
        json.dump(cfg.serialize, fp)

    # Save the command
    with open(args.out_path + '/command.txt', 'w') as f:
        f.write(' '.join(sys.argv))


    #backup src
    if cfg.dataset=='pedestrians':
        copy_src(args.root_path, os.path.dirname(args.out_path) + '/src/', args.overwrite)
        copy_src('../../experiments/', os.path.dirname(args.out_path) + '/experiments/', args.overwrite)
    else:
        copy_src(args.root_path, args.out_path + '/src/', args.overwrite)
        copy_src('../../experiments/', args.out_path + '/experiments/', args.overwrite)

    return args


def leapfrogIntegrator(last_pos, pred_states, dt=0.4):

    # if 'pos' in pred_states:
    #     pred_pos = pred_states['pos'][0]

    if 'rel' in pred_states:
        pred_pos = last_pos.unsqueeze(1).unsqueeze(0) + pred_states['rel'][0]

    elif 'vel' in pred_states:
        pred_pos = last_pos.unsqueeze(1).unsqueeze(0) + dt*pred_states['vel'][0].cumsum(2)

    if 'acc' in pred_states:
        pred_pos += 0.5*pred_states['acc'][0].cumsum(2)*dt**2

    return pred_pos


def get_shape_of_dict_items(dict, ):
    shape = {s:v.shape for s, v in dict.items()}
    # print(shape)
    return shape

def quantize_tensor(x, q=None):
    if q is None:
        return x
    return q * torch.round(x/q)

def quantize_array(x, q=None):
    if q is None:
        return x
    return q * np.round(x/q)

def optimize(cfg, loss, optimizer):

    #zero grad
    optimizer.zero_grad()

    #compute gradients
    loss.backward(retain_graph=False) 

    #clip norm
    if cfg.grad_clip is not None:
        # sdist = torch.distributions.studentT.StudentT(5)
        # for p in optimizer.param_groups[0]['params']:
        #     # pass
        #     if p.grad is not None:
        #         p.grad = p.grad + sdist.rsample(p.shape).to(device)

        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], cfg.grad_clip)          
    
    #update
    optimizer.step()