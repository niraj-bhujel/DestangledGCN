#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:01:47 2021

@author: loc
"""

import os
import math
import random
import numpy as np
from copy import copy
from collections import defaultdict
from packaging import version as pkg_version
import pandas as pd

import torch
import dgl

from data.states import NODE_TYPES


def graph_data_to_df(g, node_attrs=None):

    if node_attrs is None:
        node_attrs =  [s for s,_ in g.node_attr_schemes().items()]

    ndata = {s:g.ndata[s] for s in node_attrs}
    ndata = {s:data.unsqueeze(1) if len(data.shape)==1 else data for s, data in ndata.items()}

    labels = [(f'{k}_x', f'{k}_y') if data.shape[1]==2 else (k, ) for k, data in ndata.items()]

    ndata_flat = torch.cat([data for s, data in ndata.items()], dim=-1).cpu().numpy()
    labels_flat = [l for label in labels for l in label]

    return pd.DataFrame(data=ndata_flat, columns=labels_flat)

def get_ndata(g, node_attr, tids=None):
    if tids is None:
        tids = g.ndata['tid'].unique()
    return [g.ndata[node_attr][g.ndata['tid']==tid] for tid in tids]

def ndata_to_traj(g, ndata, tids=None):
    if tids is None:
        tids = g.ndata['tid'].unique()
    return [ndata[g.ndata['tid']==tid].cpu().numpy() for tid in tids]

def edata_from_ndata(g, node_pos):
    
    '''
    ndata : Tensor arrray of shape (num_nodes, 2) or (num_samples, num_nodes, 2) indicating the node pos
    Return 
    edata: (1, num_edges, dim) or (num_samples, num_edges, dim)
    '''

    src_edges, dst_edges = g.edges()

    if node_pos.dim()==3:
        src_pos = node_pos[:, src_edges, :]
        dst_pos = node_pos[:, dst_edges, :]
        spatial_mask = g.edata['spatial_mask'].unsqueeze(0).repeat(node_pos.shape[0], 1, 1)

    elif node_pos.dim()==2:
        src_pos = node_pos[src_edges, :]
        dst_pos = node_pos[dst_edges, :]    
        spatial_mask = g.edata['spatial_mask']

    diff = dst_pos - src_pos
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True)

    edata = {}

    edata['diff'] = diff
    edata['dist'] = dist
    edata['dir'] = torch.where(dist>0, diff/dist, diff)

    edata['temporal_mask'] =  1-spatial_mask

    edata['spatial_mask'] = spatial_mask

    return edata

def traj_to_graph_data(g, traj=None, traj_ids=None, zero_indexed=True):
    '''
    convert traj [K, N, 12, ndim] into nodes data [K, N*var_len, ndim]  to use in  g
    NOTE: Sometime throw CUDA error, if g and traj are in different device
     '''

    if traj_ids is None:
        traj_ids = g.ndata['tid'].unique()
    # map N to tid
    idx_dict = {int(v):i for i, v in enumerate(traj_ids)}
    node_idx = torch.tensor([idx_dict.get(int(tid)) for tid in g.ndata['tid']]).to(g.device)
    # node_idx = torch.tensor([idx_dict.get(int(tid)) for tid in g.ndata['tid']]).to(g.device)
    # node_idx = torch.nonzero(g.ndata['tid'][..., None]==traj_ids)[:, 1]
    time_idx = g.ndata['ntx'].flatten().type(torch.int64)
    if zero_indexed:
        time_idx = time_idx - time_idx.min() #zero indexed time steps of the traj
        
    # access row from ty_pred corresponding to each trajectories indicated by the traj_node_idx
    if traj is not None:
        traj = traj[..., node_idx, time_idx, :]
        # return , node_idx, time_idx

    return traj, node_idx, time_idx

def graph_data_to_traj(g, ndata, seq_len, traj_ids=None, fill_value=float(0.)):
    '''
    ndata: tensor [K, num_nodes, ndim] or [num_nodes, dim]
    Return:
    traj_seq: tensor of shape [K, N, seq_len, dim] or [N, seq_len, dim]
    '''
    if traj_ids is None:
        traj_ids = g.ndata['tid'].unique()
    
    if ndata.dim()==2:
        
        target_shape = (len(traj_ids), seq_len, ndata.shape[-1])
        
        traj_list = [ndata[g.ndata['tid']==tid, :] for tid in traj_ids]

    elif ndata.dim()==3:
        
        target_shape = (ndata.shape[0], len(traj_ids), seq_len, ndata.shape[-1])
        
        traj_list = [ndata[:, g.ndata['tid']==tid, :] for tid in traj_ids] # list of N tensors with each elem [K, N*T, dim]
    
    # zero indexed
    traj_ntx = [g.ndata['ntx'].flatten()[g.ndata['tid']==tid] for tid in traj_ids]
    traj_ntx = [ntx-g.ndata['ntx'].flatten().min() for ntx in traj_ntx]
    traj_ntx = [ntx.type(torch.int64) for ntx in traj_ntx]

    traj_seq = torch.full(size=target_shape, fill_value=fill_value, dtype=torch.float).to(g.device)
    traj_mask = torch.full(size=target_shape[:-1], fill_value=1, dtype=torch.float).to(g.device) # 1 indicate no gt

    for i in range(len(traj_list)):
        traj_seq[..., i, traj_ntx[i], :] = traj_list[i]
        traj_mask[..., i, traj_ntx[i]] = 0

    return traj_seq, traj_mask

def edata_between_nodes(g, src, dst, ntype='nid', edata='dir'):

    src_nodes, dst_nodes = g.edges()
    edge_ids = {(int(u), int(v)):g.edge_ids(u, v).item() for u, v in zip(src_nodes, dst_nodes)}

    src_nodes = g.nodes()[g.ndata[ntype]==src]
    dst_nodes = g.nodes()[g.ndata[ntype]==dst]

    neighbors_eids = []
    for u in src_nodes:
        u = int(u)
        for v in dst_nodes:
            v = int(v)
            if (u, v) in edge_ids:
                neighbors_eids.append(edge_ids[(u, v)])

            # if (v, u) in edge_ids:
            #     neighbors_eids.append(edge_ids[(v, u)])

    return g.edata[edata][neighbors_eids], neighbors_eids

def get_neighbors_eids(g, traj_ids=None, time_step_edges=False):
    ''' 
    Get edge id (incoming) of the neighbors of the nodes of the traj id
    When only_spatial_eges and time_step_edges are both True, first perform spatial_masking and compute edges per time step.
    Note: this is the fastest method
    '''
    if traj_ids is None:
        traj_ids = g.ndata['tid'].unique()
    
    # if not torch.is_tensor(traj_ids):
    #     traj_ids = torch.tensor(traj_ids)
        
    src_nodes, dst_nodes = g.in_edges(g.nodes())
    edge_ids = g.edge_ids(src_nodes, dst_nodes)
    edge_mask = g.edata['spatial_mask'].squeeze(1)
    
    spatial_eids = []
    temporal_eids = []
    
    for tid in traj_ids:
        traj_nodes = g.nodes()[g.ndata['tid']==tid]
        # traj nodes index in dest_nodes
        matching_idx = torch.nonzero(dst_nodes[..., None]==traj_nodes, as_tuple=False)  # traj_nodes are destination nodes
        # 1st column is the index of the traj_nodes in the dst nodes, 2nd comumn is it's index, whieh can be the time step 
        dst_node_idx, traj_node_idx = matching_idx[:, 0], matching_idx[:, 1]
        
        # incoming eids to the nodes of current traj
        traj_eids = edge_ids[dst_node_idx] # includes eids of spatial neighbors nodes and own nodes from previous/future time step 
        traj_emask = edge_mask[traj_eids] # array of 1 or 0 , 1 reprsents spatial edge, 0 temporal edge
             
        traj_spatial_eids = traj_eids[traj_emask==1]
        traj_temporal_eids = traj_eids[traj_emask==0]
    
        if time_step_edges:
            spatial_node_idx = traj_node_idx[traj_emask==1]
            temporal_node_idx = traj_node_idx[traj_emask==0]
            
            traj_spatial_eids = [traj_spatial_eids[spatial_node_idx==t] for t in range(len(traj_nodes))]
            traj_temporal_eids = [traj_temporal_eids[temporal_node_idx==t] for t in range(len(traj_nodes))]
        
        spatial_eids.append(traj_spatial_eids)
        temporal_eids.append(traj_temporal_eids)
                
    return spatial_eids, temporal_eids
        
def get_neighbors_edata(g, traj_ids, edge_data=None, edge_feature='dist', aggregation='average', default_value=0):
    '''For each traj_ids, get  neighbors edge data (mean influences)
    Return: tensor of shape [N, edge_dim]
    '''

    if not torch.is_tensor(traj_ids):
        traj_ids = torch.tensor(traj_ids)
        
    if edge_data is None:
        edge_data = g.edata[edge_feature]
    
    # start = time.time()
    neighbors_eids, _  = get_neighbors_eids(g, traj_ids, time_step_edges=False)
    
    default_edata = torch.full_like(edge_data, fill_value=default_value)[:1, :] # (1, edim)

    if aggregation=='max':
        neighbors_data = [torch.max(edge_data[eids], dim=0, keepdim=True)[0] if len(eids)>0 else default_edata for eids in neighbors_eids]
    if aggregation=='min':
        neighbors_data = [torch.min(edge_data[eids], dim=0, keepdim=True)[0] if len(eids)>0 else default_edata for eids in neighbors_eids]
    elif aggregation=='average':
        neighbors_data = [torch.mean(edge_data[eids], dim=0, keepdim=True) if len(eids)>0 else default_edata for eids in neighbors_eids]
    elif aggregation =='sum':
        neighbors_data = [torch.sum(edge_data[eids], dim=0, keepdim=True) if len(eids)>0 else default_edata for eids in neighbors_eids]
    else:
        raise Exception(f'Aggregration method {aggregation} for edge pooling is not valid!!')
        
    neighbors_data = torch.cat(neighbors_data, dim=0)
    
    # # check for nan
    # if torch.isnan(neighbors_data):
    #     raise Exception (f'NaN values in neighbors_data {neighbors_data}')
    
    neighbors_data[neighbors_data!=neighbors_data]=default_value
    
    # print('time required:', time.time() - start)
    return neighbors_data, neighbors_eids

def get_neighbors_step_edata(g, traj_ids, edge_data=None, edge_feature='dist'):
    ''' 
    For each tid, get the edge id with respect for each neighbors traj (tid) at each time step (faster than get_neighbors_edata, when num_ped is large)
    The final values of the neighbors data is same as get_neighbors_edata. Use this function only when neighbors data at each 
    time step is required.  
    Return: tensor of shape [T, N, edge_dim]
    '''

    if edge_data is None:
        edge_data = g.edata[edge_feature]
        
    traj_nodes = [g.nodes()[g.ndata['tid']==tid] for tid in traj_ids]
    traj_in_eids = [g.in_edges(nodes, form='eid').cpu().numpy() for nodes in traj_nodes]
    traj_out_eids = [g.out_edges(nodes, form='eid').cpu().numpy() for nodes in traj_nodes]
    
    temporal_eids = []
    spatial_eids = []
    neighbors_edata = torch.zeros(len(traj_ids), len(traj_ids), edge_data.size()[-1]).to(g.device)
    
    for t, tid in enumerate(traj_ids):
        # temporal edges
        t_t_eids = np.intersect1d(traj_in_eids[t], traj_out_eids[t]) 
        temporal_eids.append(t_t_eids)
        # temporal edata
        # if len(t_t_eids)>0:
        #     neighbors_edata[t, t, :] = edge_data[t_t_eids].mean()            
        
        t_spatial_eids = []
        for n, nid in enumerate(traj_ids):
            if t==n:
                continue
            # common edges between the in edges and out edges of traj form neighborhood
            t_n_eids = np.intersect1d(traj_in_eids[t], traj_out_eids[n])
            t_spatial_eids.append(t_n_eids)

            if len(t_n_eids)>0:
                neighbors_edata[t, n,  :] = edge_data[t_n_eids].mean()
        
        spatial_eids.append(t_spatial_eids)
        
    return neighbors_edata, spatial_eids, temporal_eids


def get_lane_nodes(g, traj_ids=None, lane_idx=2):
    if traj_ids is None:
        traj_ids = g.ndata['tid'][g.ndata['cid']!=3].unique()
        
    lane_nodes = {}
    
    for i, tid in enumerate(traj_ids):
        traj_nodes = g.nodes()[g.ndata['tid']==tid]

        # incoming edges
        src_nodes, _ = g.in_edges(traj_nodes)
        
        src_lane_nodes = src_nodes[g.ndata['cid'][src_nodes]==lane_idx]

        # check for multiple lanes, if multiple, choose 1 randomly
        lane_tids = g.ndata['tid'][src_lane_nodes].unique()
        if len(lane_tids)>1:
            src_lane_nodes = g.nodes()[g.ndata['tid']==random.choice(lane_tids)]

        lane_nodes[int(tid)] = src_lane_nodes
        
    return lane_nodes

def filter_node_types(g, keep_node_types, node_type_list):
    
    node_cids = [node_type_list.index(node) for node in keep_node_types]
    
    # filter node types
    if pkg_version.parse(torch.__version__) > pkg_version.parse("1.10"):
        keep_nodes = torch.isin(g.ndata['cid'], torch.tensor(node_cids))
    else:
        keep_nodes = torch.cat([g.nodes()[g.ndata['cid']==cid] for cid in node_cids]).unique()
    
    return dgl.node_subgraph(g, keep_nodes)
    
def remove_node_type(g, include_nodes, node_type_list):
    ''' remove nodes that are not in the include_nodes'''
    
    nodes_to_removed = []
    for node_type in include_nodes:
        cid = node_type_list.index(node_type)
        nodes_to_removed.append(g.nodes()[g.ndata['cid']!=cid])
    return dgl.remove_nodes(g, torch.cat(nodes_to_removed))
        

def filter_nodes(g, min_len=2, min_edges=1):
    ''' remove nodes of trajectory with length less than min_len and edges less than min_edges'''
    # assues g.nodes() and g.ndata['tid'] correspond each other
    removed_nodes = []
    for node, tid in zip(g.nodes(), g.ndata['tid']):
        if len(g.ndata['ntx'].flatten()[g.ndata['tid']==tid])<min_len:
            removed_nodes.append(node)
        if (g.in_degrees(node) + g.out_degrees(node))<1:
            removed_nodes.append(node)
    g = dgl.remove_nodes(g, removed_nodes)
    return g

def remove_redundant_nodes(trgt_graphs, obsv_graphs):
    ''' remove the nodes from target graphs that are not in obsv_graphs'''
    refined_trgt_graphs = []
    for gx, gt in zip(dgl.unbatch(obsv_graphs), dgl.unbatch(trgt_graphs)):
        obsv_traj_id = gx.ndata['tid'].unique().cpu().numpy()
        trgt_traj_id = gt.ndata['tid'].unique().cpu().numpy()
        comm_traj = np.sort(np.intersect1d(obsv_traj_id, trgt_traj_id))
        # target trajectories that are not in the observed graph
        redundant_trgt_tid = np.setdiff1d(trgt_traj_id, comm_traj)
        # print(redundant_trgt_tid)
        if len(redundant_trgt_tid)>0:
            # get the node number for redundant traj
            redundant_trgt_nid = [gt.ndata['nid'][gt.ndata['tid']==tid] for tid in redundant_trgt_tid]
            #zero indexed node
            redundant_trgt_nid = torch.cat(redundant_trgt_nid) - torch.min(gt.ndata['nid'])
            # remove nodes/trajectory  from target graphs that is not in the observed graphs
            gt = dgl.remove_nodes(gt, redundant_trgt_nid)
        
        refined_trgt_graphs.append(gt)
    return dgl.batch(refined_trgt_graphs, list(trgt_graphs.node_attr_schemes()), list(trgt_graphs.edge_attr_schemes()))
    
def get_nodes_pos_at_t(g, t=7):
    '''
    Get the node pos at the time step. If node pos doesn't exist at the given step, return first pos   
    '''
    traj_full_pos = {tid:g.ndata['pos'][g.ndata['tid']==tid] for tid in g.ndata['tid'].unique().numpy()}
    traj_time_steps = {tid:g.ndata['ntx'].flatten()[g.ndata['tid']==tid] for tid in g.ndata['tid'].unique().numpy()}
    
    nodes_pos_at_t = []
    for tid in g.ndata['tid'].numpy():
        node_time_steps = traj_time_steps[tid]

        if t in node_time_steps:
            init_pos_idx = (node_time_steps==t).nonzero(as_tuple=False)[0][0]
        else:
            init_pos_idx = 0

        nodes_pos_at_t.append(traj_full_pos[tid][init_pos_idx])

    return torch.stack(nodes_pos_at_t, dim=0)

def split_graph(g, split_idx=7):
    ''' split graph into observed into target graphs '''
    g1_nodes = g.nodes()[g.ndata['ntx'].flatten()<=split_idx] # obsv_graphs
    g2_nodes = g.nodes()[g.ndata['ntx'].flatten()>split_idx] # target graphs
    g1 = dgl.node_subgraph(g, nodes=g1_nodes.type(torch.int64), store_ids=False)
    g2 = dgl.node_subgraph(g, nodes=g2_nodes.type(torch.int64), store_ids=False)    
    return g1, g2

def latent_to_graph_data(z, a, g, comm_traj):

    _, node_idx, time_idx = traj_to_graph_data(g, traj_ids=comm_traj)        

    if z.dim()==4:

        nodes_data = z[:, time_idx, node_idx, :]
        
        spatial_eids, temporal_eids = get_neighbors_eids(g, comm_traj, time_step_edges=True)
        spatial_idx = torch.tensor([[t, i]for i, step_eids in enumerate(spatial_eids) for t, eids in enumerate(step_eids) for _ in eids])
        temporal_idx = torch.tensor([[t, i]for i, step_eids in enumerate(temporal_eids) for t, eids in enumerate(step_eids) for _ in eids])
        
        temporal_edata = z[:, temporal_idx[:, 0], temporal_idx[:, 1], :]
        
        if spatial_idx.size(0)>0:
            spatial_edata = a[:, spatial_idx[:, 0], spatial_idx[:, 1], :]
            edge_data = torch.cat([spatial_edata , temporal_edata], dim=1)
        else:
            edge_data = temporal_edata

        # reorder edges
        eids_flat = torch.cat([torch.cat(eids) for eids in spatial_eids] + [torch.cat(eids) for eids in temporal_eids])
        edge_data = edge_data[:, eids_flat.sort().indices, :]
    
    else:
        nodes_data = z[:, node_idx, :]
        # incoming spatial edge and temporal edge ids of each node at each time step
        spatial_eids, temporal_eids = get_neighbors_eids(g, comm_traj, time_step_edges=False)
        
        # node idx of each spatial edges
        spatial_idx = torch.tensor([i for i, step_eids in enumerate(spatial_eids) for _ in step_eids])
        
        # node idx for each node temporal edges
        temporal_idx = torch.tensor([i for i, step_eids in enumerate(temporal_eids) for _ in step_eids])
        
        temporal_edata = z[:, temporal_idx, :]
        
        # may not always have spatial edges
        if spatial_idx.size(0)>0:
            spatial_edata = a[:, spatial_idx, :]
            edge_data = torch.cat([spatial_edata, temporal_edata], dim=1)
        
        else:
            edge_data = temporal_edata
        
        # order edge data, edge_id also represent index in edge data as edge_id numbering start from 0
        eids_flat = torch.cat(spatial_eids+temporal_eids) # ordering of concatenation must be same as edge_data concatatenation
        edge_data = edge_data[:, eids_flat.sort().indices, :]
        
    return node_idx, time_idx, nodes_data, edge_data


def no_edges(graphs_list):
    edges = [g.number_of_edges()>1 for g in graphs_list]
    return not np.all(edges)

