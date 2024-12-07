#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:02:16 2022

@author: dl-asoro
"""
import math
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import copy
from data.tree import Node, get_all_paths
        
def lines_in_patch(node_coords, patch, pad=0, mode='intersect'):
    '''
    node_coords : lines (N, 2)
    patch : (x_min, y_min, x_max, y_max).
    Returns
    -------
    True: if a point inside patch else False
    '''

    x_min, y_min, x_max, y_max = patch
    
    cond_x = np.logical_and(node_coords[:, 0] < x_max+pad, node_coords[:, 0] > x_min-pad)
    cond_y = np.logical_and(node_coords[:, 1] < y_max+pad, node_coords[:, 1] > y_min-pad)
    cond = np.logical_and(cond_x, cond_y)
    if mode == 'intersect':
        return np.any(cond)
    elif mode == 'within':
        return np.all(cond)

def trim_line(line, xy_min, xy_max):
    valid_idxs = np.logical_and(line>xy_min, line<xy_max).all(-1)
    return line[valid_idxs]

def interpoalte_points(points, num_points, kind='linear'):
    interp = interp1d(np.arange(len(points)), points, kind=kind)
    return np.array([interp(i) for i in np.linspace(0, len(points)-1, num=num_points)])
    
def interpolate_line(line, num_points, kind='linear'):

    interp_x = interp1d(np.arange(len(line)), line[:, 0], kind=kind)
    interp_y = interp1d(np.arange(len(line)), line[:, 1], kind=kind)
    
    return np.array([[interp_x(i), interp_y(i)] for i in np.linspace(0, len(line)-1, num=num_points)])

def interpolate_lane(lane_pos, resolutions=0.5):
    path_length = np.sqrt(np.sum(np.diff(lane_pos, axis=0) ** 2, axis=1)).sum()
    npoints = int(max(math.ceil(path_length / resolutions) + 1.5, 2)) # 0.1 resolutions

    return interpolate_line(lane_pos, npoints)

def sample_lane_points(lane_pos, num_points=12):
    if len(lane_pos)<num_points:
        lane_pos = interpolate_line(lane_pos, num_points)
        return lane_pos
    
    sample_points = np.linspace(0, len(lane_pos)-1, num=num_points, endpoint=True, dtype=np.int)
    
    return lane_pos[sample_points]

def trim_lanes_poses(lane_poses, xy_min, xy_max, pad=0):
    

    xy_min = np.array(xy_min) - pad
    xy_max = np.array(xy_max) + pad
    
    trimmed_lane_poses = {}
    # lane_poses = {lane:lane_pos[np.logical_and(lane_pos>min_values, lane_pos<max_values).all(-1)] for lane, lane_pos in curr_lane_poses.items() if len(lane_pos)>0}
    for lane_token, lane_pos in lane_poses.items():

        trimmed_lane_pos = trim_line(lane_pos, xy_min, xy_max)
        
        if len(trimmed_lane_pos)>0:
            trimmed_lane_poses[lane_token] = trimmed_lane_pos
    
    return trimmed_lane_poses

def closest_lane_points(lane_pos, node_pos, min_dim=0, resolutions=0.5):
    '''
    Get the closest point in the lane pos from the trajectory
    lane_pos: array of shape (L, 2)
    node_pos: array of shape (T, 2), node trajectory
    '''
    if resolutions!=0.5:
        lane_pos = interpolate_lane(lane_pos, resolutions)
    
    closest_idxs = euclidean_distances(lane_pos, node_pos).argmin(min_dim)
    
    return lane_pos[closest_idxs]

def get_closest_lane(node_pos, lane_poses, d_thresh=10, check_heading=True, metric='euclidean', mode='start_min_max'):
    '''get closet lane based on distance from node pos'''
    d_closest = np.inf
    closest_lane = None
    
    node_heading = np.sign(node_pos[-1] - node_pos[0] + 1e-6)
    for lane, lane_pos in lane_poses.items():
        
        if check_heading:
            lane_heading = np.sign(lane_pos[-1] - lane_pos[0])
            if np.any(node_heading!=lane_heading):
                continue
            
        pairwise_dist = pairwise_distances(node_pos, lane_pos, metric=metric) #[len(node_pos), len(lane_pos)]

        if mode=='start_min_max':
            d_min = pairwise_dist[0].min()
            d_max = pairwise_dist[0].max()
            
        # min distance from end node_pos
        if mode=='end_min_max':
            d_min = pairwise_dist[-1].min()
            d_max = pairwise_dist[-1].max()
        # averge of min distance from node_pos            
        elif mode=='avg_min_max':
            d_min = pairwise_dist.min(-1).mean()
            d_max = pairwise_dist.max(-1).mean()
        else:
            d_min = pairwise_dist.min()
            d_max = pairwise_dist.max()
            
        if d_max>d_thresh:
            continue
        
        if d_min<d_closest:
            d_closest = d_min
            closest_lane = lane
            
    return closest_lane

def nearest_node_lanes(node_pos, lane_poses, max_angle=60, trim_lanes=False, d_thresh=3):
    ''' find lanes that are closest to node_pos based on closest distances'''
    
    # trim lanes before finding closest points, useful for calculating distances only from closest point
    if trim_lanes:
        lane_poses = trim_lanes_poses(copy(lane_poses), node_pos.min(0), node_pos.max(0), pad=5)

    node_heading = node_pos[-1] - node_pos[-2] 
    node_norm = np.linalg.norm(node_heading)
    if node_norm>0:
        node_heading = node_heading/node_norm

    nearest_lanes = []
    nearest_dist = []
    nearest_lane = None
    d_min = np.inf
    for lane, lane_pos in lane_poses.items():
        
        lane_heading = lane_pos[-1] - lane_pos[0]
        norm = np.linalg.norm(lane_heading)
        if norm>0:
            lane_heading = lane_heading/norm
            
        angle = np.arccos(np.dot(node_heading, lane_heading)) * 180/np.pi
        
        if angle > max_angle:
            continue
                    
        # if check_heading_type=='all':
        #     same_headings = np.all(node_heading==lane_heading)
        
        # elif check_heading_type=='any':
        #     same_headings = np.any(node_heading==lane_heading)

        # if not same_headings:
        #     continue

        d = euclidean_distances(node_pos, lane_pos)

        # find closest from recent pos
        d_nearest = d[-1].min()

        # if d.min(0).min()<d_thresh:
        if d_nearest < d_thresh:
            nearest_lanes.append(lane)
            nearest_dist.append(d_nearest)

        if d_nearest<d_min:
            d_min = d_nearest
            nearest_lane = lane
    
    nearest_lanes = [nearest_lanes[idx] for idx in np.argsort(nearest_dist)]

    return nearest_lanes, nearest_lane


def lanes_along_node(curr_lane, node_pos, lane_poses, nusc_map, radius=10, max_dist=3):
    '''Given node poses (num_nodes, 2) and lane_poses (dict), find all lanes closed to the nodes using outgoing lanes'''
        
    # starting lane from the initial node_pos
    # curr_lane = get_closest_lane(lane_poses, node_pos, mode='avg_min')
    

    # _, curr_lane = nearest_node_lanes(node_pos, lane_poses, trim_lanes=True)
    
    # curr_lane = nusc_map.get_closest_lane(node_pos[0][0], node_pos[0][1], radius=radius) # reliable for starting point instead of last 
    
    # if not curr_lane:
    #     return None, None
    
        #%%
        
    node_lanes = []
    node_lanes.append(curr_lane)
    

    lane_tree = Node(data=None)
    child = Node(curr_lane)
    lane_tree.addNode(child)
    while curr_lane:
        
        outgoing_lanes = nusc_map.get_outgoing_lane_ids(curr_lane) # list of lane tokens
        outgoing_lanes = [l for l in outgoing_lanes if l in lane_poses]
        outgoing_lanes_pos = {l:lane_poses[l] for l in outgoing_lanes}
        # neighbor_lanes, nearest_lane= nearest_node_lanes(node_pos, {l:lane_poses[l] for l in outgoing_lanes}, check_heading=False, trim_lanes=False, d_thresh=max_dist)
        # curr_lane = nearest_lane if nearest_lane in neighbor_lanes else None
        curr_lane = get_closest_lane(node_pos, outgoing_lanes_pos, d_thresh=max_dist, check_heading=True if len(outgoing_lanes)>1 else False) 
                
        for lane in outgoing_lanes:
            # if lane in neighbor_lanes:
            child.addNode(Node(lane))
            node_lanes.append(lane)

        for node in child.children:
            if node.data==curr_lane:
                child = node
        
        if len(node_lanes)>10:
            break

    node_paths = get_all_paths(lane_tree)
    #%%    
    return node_lanes, node_paths
                            
def plot_node_lanes(lane_poses, node_lanes_dict, node_poses_list, patch=None, plot_all_lanes=False):
    """
    lane_poses : List of lanes, each with shape (lane_len, 2)
    patch : (x_min, y_min, x_max, y_max)
    node_poses : List of traj, each with shape (node_len, 2)
    """
    fig, ax = plt.subplots()
    
    colors = np.random.random((len(node_poses_list), 3))
    for i, node_pos in enumerate(node_poses_list):
        xs, ys = node_pos[:, 0], node_pos[:, 1]
        # color = np.random.random((3,))
        plt.plot(xs, ys, color=colors[i], linestyle='dashed', marker='*', markersize=4, label=i, zorder=500)
        plt.plot(xs[0], ys[0], color=colors[i], marker='s', markersize=6, zorder=500)
    
    for i, lanes in node_lanes_dict.items():
        for lane in lanes:
            if lane is not None:
                xs, ys = lane_poses[lane][:, 0], lane_poses[lane][:, 1]
                plt.plot(xs, ys, color=colors[i], linestyle='solid', zorder=300)

    if patch is not None:
        rect = Rectangle((patch[0], patch[1]), patch[2]-patch[0], patch[3]-patch[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    if plot_all_lanes:
        # plot all lanes
        for _, lane_pos in lane_poses.items():
            plt.plot(lane_pos[:, 0], lane_pos[:, 1], linestyle='solid', zorder=8)
        
    plt.legend()
    plt.show()
    return fig

def plot_lanes(lane_poses, node_lanes, node_pos=None, patch=None, legend=False, title=''):
    fig, ax = plt.subplots()
    for lane in node_lanes:
        lane_pos = lane_poses[lane]
        plt.plot(lane_pos[:, 0], lane_pos[:, 1], linestyle='solid', zorder=800,)
    
    if node_pos is not None:
        color = np.random.random((3,))
        plt.plot(node_pos[:, 0], node_pos[:, 1], color=color, linestyle='dashed', marker='*', markersize=4, zorder=500)
        plt.plot(node_pos[0, 0], node_pos[0, 1], color=color, marker='s', markersize=6, zorder=500)
            
    if patch is not None:
        rect = Rectangle((patch[0], patch[1]), patch[2]-patch[0], patch[3]-patch[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(title)
    if legend:
        plt.legend()
    plt.show()
    return fig
    
    

