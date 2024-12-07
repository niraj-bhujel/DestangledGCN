import math

import dgl
import torch
import numpy as np
import random

from copy import copy
from scipy.interpolate import interp1d
from collections import defaultdict

from data.states import *

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    vec_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    vec_norm[vec_norm == 0.0] = 1.0
    return vector/vec_norm

def angle_between(v1, v2, vis=False, degree=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # if vis:
    #     origin = np.array([[0, 0],[0, 0]])
    #     V = np.stack([v1, v2])
    #     plt.figure(num='Vector_angle')
    #     plt.quiver(*origin, V[:, 0], V[:, 1], color=['r','b'], scale=5)
    #     plt.title("Red:V1, Blue: V2")
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if degree:
        angle = np.degrees(angle)
    return angle

def field_of_view(nodes_dir, edir, twophi=200.0):
    cosphi = math.cos(twophi / 2.0 / 180.0 * math.pi)
    cosphi_l = np.einsum('aj, abj->ab', nodes_dir, edir)
    # phi_l = np.arccos(cosphi_l)*180/math.pi
    in_sight = cosphi_l > cosphi

    return in_sight.astype(np.int)

def constant_velocity_accleration(init_pos, vel, acc, pred_steps, time_step=0.5):
    preds = []
    x, y = init_pos
    vx, vy = vel
    ax, ay = acc
    sec_from_now = np.round(pred_steps * time_step, 1)
    for t in np.arange(time_step, sec_from_now + time_step, time_step):
        half_time_squared = 0.5 * t * t
        preds.append((x + t * vx + half_time_squared * ax,
                      y + t * vy + half_time_squared * ay))

    if isinstance(init_pos, torch.Tensor):
        return torch.stack([torch.stack(p) for p in preds])
    else:
        return np.array(preds)
    return preds

def constant_velocity(init_pos, vel, acc=None, pred_steps=12, time_step=0.5):
    
    preds = []
    x, y = init_pos
    vx, vy = vel
    sec_from_now = np.round(pred_steps * time_step, 1)
    for t in np.arange(time_step, sec_from_now + time_step, time_step):
        preds.append((x + t * vx, y + t * vy))

    if isinstance(init_pos, torch.Tensor):
        return torch.stack([torch.stack(p) for p in preds])

    else:
        return np.array(preds)
                    
def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0., 0., 0.
    return (path_length / path_distance) - 1, path_length, path_distance


def smooth_points(points, method='quadratic'):
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2 +1e-6, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # if not sorted, cannot interpoalte
    if np.any(distance[1:] <= distance[:-1]):
        return points
    
    interpolator =  interp1d(distance, points, kind=method, axis=0)

    return interpolator(np.linspace(0, 1, len(points)))

def get_direction(direction_vector):
    
    norm = np.linalg.norm(direction_vector, ord=2)

    if norm>0:
        return direction_vector/norm
    else:
        return direction_vector

def motion_kinematics(p, dt, obsv_idx):

    v = np.gradient(p, dt, axis=0)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    
    v_dir = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 0.))
    
    if len(p[:obsv_idx])>2:
        speed = np.linalg.norm((p[1:obsv_idx] - p[:obsv_idx-1])/dt, axis=-1).mean()
    else:
        speed = np.linalg.norm((p[1] - p[0])/dt, axis=-1)

    a = np.gradient(v, dt, axis=0)
    a_norm = np.linalg.norm(a, axis=-1, keepdims=True)

    r = p - p[0] #NOTE! When start position is change, r_dir also changes 
    # r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    # r_dir = np.divide(r, r_norm, out=np.zeros_like(r), where=(r_norm > 0.))

    # pred_pos = constant_velocity(p[obsv_idx], v[obsv_idx], len(p[obsv_idx+1:]), time_step=dt)

    goal = p[-1]

    gdir = goal - p
    gnorm = np.linalg.norm(gdir, axis=-1, keepdims=True)

    h = np.divide(gdir, gnorm, out=np.zeros_like(gdir), where=(gnorm > 0.))

    return p, r, v, v_norm, v_dir, a, a_norm, h, goal, speed

def node_sequence(node_pos, dt, pad_front=0, pad_end=20, seq_len=20, obsv_idx=0, frames=None, yaw=None):
    
    p, r, v, v_norm, v_dir, a, a_norm, h, goal, speed = motion_kinematics(node_pos, dt, obsv_idx)
        
    state_dict = {}
    for s in ['pos', 'rel', 'vel', 'acc', 'dir', 'hed', 'goal']:
        state_dict[s] = np.zeros((seq_len, 2))
        
    for s in ['rnorm', 'vnorm', 'anorm', 'yaw', 'speed']:
        state_dict[s] = np.zeros((seq_len, 1))

    for s in ['msk', 'fid']:
         state_dict[s] = np.zeros((seq_len,))
    
    state_dict['msk'][pad_front:pad_end] = 1        
    state_dict['pos'][pad_front:pad_end] = p
    
    state_dict['rel'][pad_front:pad_end] = r
    # state_dict['rnorm'][pad_front:pad_end] = r_norm
    
    state_dict['vel'][pad_front:pad_end] = v
    state_dict['vnorm'][pad_front:pad_end] = v_norm

    state_dict['acc'][pad_front:pad_end] = a
    state_dict['anorm'][pad_front:pad_end] = a_norm

    state_dict['dir'][pad_front:pad_end] = v_dir

    state_dict['hed'][pad_front:pad_end] = h
    state_dict['goal'][pad_front:pad_end] = goal

    state_dict['speed'][pad_front:pad_end] = speed

    if yaw is None:
        yaw = np.arctan2(v[:, 1], v[:, 0]).reshape(-1, 1)

    state_dict['yaw'][pad_front:pad_end] = yaw

    if frames is not None:
        state_dict['fid'][pad_front:pad_end] = frames
        
    return state_dict


def get_interaction_radius(cid1, cid2, interaction_radius):
    try:
        idist = interaction_radius[NODE_TYPES[int(cid1)], NODE_TYPES[int(cid2)]]
    except Exception:
        try:
            idist = interaction_radius[NODE_TYPES[int(cid2)], NODE_TYPES[int(cid1)]]
        except Exception as e:
            print(e)
            raise Exception('Unable to find idist between {NODE_TYPES[int(cid2)]} and {NODE_TYPES[int(cid2)]}')
    return idist

def norm(p1, p2): 
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) + 1e-6
        
def seq_to_st_graph(sequence_dict, interaction_radius, seq_len=20, drop_sedge=0.0, twophi=200):

#%% 
    for key, val in sequence_dict.items():
        if key in NODE_INFO:
            sequence_dict[key] = np.array(val) 
        else:
            sequence_dict[key] = np.stack(val, 0)

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)
    N = 0 #total nodes counter
    previous_nodes = None
    cosphi = math.cos(twophi / 2.0 / 180.0 * math.pi)
    for t in range(seq_len):
        current_mask = sequence_dict['msk'][:, t].astype(bool) # indicates nodes present at current time
        current_nodes = [n for n in range(N, N + current_mask.sum())]
        if not current_nodes:
            continue

        current_states = {s:sequence_dict[s][:, t][current_mask] for s in NODE_STATES[1:]}
        current_info = {s:sequence_dict[s][current_mask] for s in NODE_INFO}
        
        # temporal edges
        unmatched_nodes = copy(current_nodes) # new nodes at current time
        if t>0 and previous_nodes is not None:
            for j, v in enumerate(current_nodes):
                for i, u in enumerate(previous_nodes):
                    if previous_info['tid'][i]!=current_info['tid'][j]:
                        continue

                    # if random.random()>0.5:
                    #     continue

                    # dist = norm(previous_pos[i], current_pos[j])
                    diff = current_states['pos'][j] - previous_states['pos'][i]  # direction is measured from edge destination
                    dist = np.linalg.norm(diff, axis=-1)
                    edir = diff/dist if dist>0 else diff

                    edges_data['src'].extend([u])
                    edges_data['des'].extend([v])

                    edges_data['dist'].extend([dist])
                    edges_data['diff'].extend([diff])
                    edges_data['dir'].extend([edir])

                    edges_data['temporal_mask'].extend([1.0])
                    edges_data['spatial_mask'].extend([0.0])
                    
                    # # bidirectional edges
                    # if random.random()>0.5:
                    #     edges_data['src'].extend([v])
                    #     edges_data['des'].extend([u])
                    #     edges_data['dist'].extend([dist])
                    #     edges_data['diff'].extend([-diff])
                    #     edges_data['spatial_mask'].extend([0.0])
                        
                    # remove current node
                    unmatched_nodes.remove(v)
                    # only single node from previous step can match to single node at current step, thus break if matched
                    break 
        # make sure that nodes at last time step is not added to the graph as it will not have any temporal edges
        if t==seq_len-1:
            for node in unmatched_nodes:
                current_nodes.remove(node)
        
        # node attributes
        for i in range(len(current_nodes)):
            u = current_nodes[i]

            # add to nodes data
            nodes_data['ntx'].append(t)
            for s in NODE_STATES[1:]:
                nodes_data[s].append(current_states[s][i])
                
            for s in NODE_INFO:
                nodes_data[s].append(current_info[s][i])
            
            # spatial edges
            for j in range(len(current_nodes)):

                v = current_nodes[j]

                if i==j:
                    continue

                if random.random()<drop_sedge:
                    continue
                
                diff = current_states['pos'][j] - current_states['pos'][i] # (v - u)
                dist = np.linalg.norm(diff, axis=-1)
                edir = diff/dist if dist>0 else diff
                
                idist = get_interaction_radius(current_info['cid'][i], current_info['cid'][j], interaction_radius)

                if dist > idist:
                    continue

                # interaction only between node and its corresponding lane NOTE: nid for vehicle and corresponding lane is same
                if (current_info['cid'][i]==NODE_TYPES.index('VEHICLE') and current_info['cid'][j]==NODE_TYPES.index('LANE')) or (current_info['cid'][i]==NODE_TYPES.index('LANE') and current_info['cid'][j]==NODE_TYPES.index('VEHICLE')):
                    if current_info['nid'][i]!=current_info['nid'][j]:
                        continue

                fov = np.dot(current_states['dir'][i], edir)
                if fov > cosphi:
                    # effect from neighbor node
                    edges_data['src'].extend([v])
                    edges_data['des'].extend([u])
                    edges_data['dist'].extend([dist])
                    edges_data['diff'].extend([-diff]) # difference measured from destination
                    edges_data['dir'].extend([-edir])
                    edges_data['spatial_mask'].extend([1.0])
                    edges_data['temporal_mask'].extend([0.0])
                            
        previous_nodes = current_nodes
        previous_states = current_states
        previous_info = current_info

        N+=len(current_nodes)
        
    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['des']))
    
    assert g.number_of_edges()==len(edges_data['src']), f"Graph created with {g.number_of_edges()} edges, but have {edges_data['src']} edge data!!"
    #%%
    # Add  features
    for attr in NODE_INFO:
        g.ndata[attr] = torch.tensor(nodes_data[attr], dtype=torch.int)

    g.ndata['dt'] = torch.tensor(nodes_data['dt'], dtype=torch.float)


    for attr in NODE_STATES:
        g.ndata[attr] = torch.tensor(np.stack(nodes_data[attr], 0).reshape(g.number_of_nodes(), STATE_DIMS[attr]), dtype=torch.float)

    for attr in EDGE_ATTRS:
        g.edata[attr] = torch.tensor(np.reshape(edges_data[attr], (g.num_edges(), STATE_DIMS[attr])), dtype=torch.float)

    return g

def seq_to_fov_graph(sequence_dict, interaction_radius=INTERACTION_RADIUS):

    nodes_data = {}
    peds_len = [len(pos) for pos in sequence_dict['pos']]
    for key, val in sequence_dict.items():
        if key in NODE_INFO:
            nodes_data[key] = np.concatenate([[v]*peds_len[i] for i, v in enumerate(val)])
        else:
            nodes_data[key] = np.concatenate(val , 0)

    peds_tid = nodes_data['tid']
    peds_ntx = nodes_data['ntx'].flatten()

    # temporal and spatial adjoint matrix 1-edge, 0-no edge
    temporal_matrix = np.logical_and(peds_ntx[None, :]==peds_ntx[:, None]+1, peds_tid[None, :]==peds_tid[:, None]).astype(int)
    neighbor_matrix = (peds_tid[None, :]!=peds_tid[:, None]).astype(int)
    sametime_matrix = (peds_ntx[None, :]==peds_ntx[:, None]).astype(int)
    futuretime_matrix = (peds_ntx[None, :]>peds_ntx[:, None]).astype(int)
    
    # consider euclidean distance
    diff = nodes_data['pos'][None, :, :] - nodes_data['pos'][:, None, :]  # dst - src, row indices-> src, col_indices->dst, 
    edir = unit_vector(diff)

    # interaction between different neighbors
    dist = np.linalg.norm(diff, axis=-1, keepdims=False)
    idist = np.zeros_like(dist)
    for i, cid1 in enumerate(nodes_data['cid']):
        for j, cid2 in enumerate(nodes_data['cid']):
            idist[i, j] = get_interaction_radius(cid1, cid2, interaction_radius)
    idist_matrix = (dist<idist).astype(int) 

    # consider different field of view
    full_fov = field_of_view(nodes_data['dir'], edir, twophi=200) # 0-1
    wide_fov = field_of_view(nodes_data['dir'], edir, twophi=120) # 0-1
    narrow_fov = field_of_view(nodes_data['dir'], edir, twophi=60)

    # consider for opposite direction
    n_dir = np.dot(nodes_data['dir'], nodes_data['dir'].T)

    # adjust fov using walking direction
    # fov_same_dir = (n_dir>0).astype(int) * wide_fov 
    fov_opp_dir = (n_dir<0).astype(int) * narrow_fov

    spatial_matrix = neighbor_matrix * np.logical_or(idist_matrix*sametime_matrix*full_fov, fov_opp_dir*idist_matrix*futuretime_matrix)
    # spatial_matrix = neighbor_matrix  * fov_opp_dir * idist_matrix * futuretime_matrix

    # final adjacenty matrix
    adj_matrix = np.logical_or(spatial_matrix, temporal_matrix)

    src_ids, dst_ids = np.where(adj_matrix>0)
    g = dgl.graph((src_ids, dst_ids))    

    # Add  features
    for attr in NODE_INFO:
        if attr=='dt':
            g.ndata[attr] = torch.tensor(nodes_data[attr], dtype=torch.float)
        else:
            g.ndata[attr] = torch.tensor(nodes_data[attr], dtype=torch.int)

    for attr in NODE_STATES:
        g.ndata[attr] = torch.tensor(np.stack(nodes_data[attr], 0).reshape(g.number_of_nodes(), STATE_DIMS[attr]), dtype=torch.float)

    edges_data = {}
    edges_data['diff'] = nodes_data['pos'][dst_ids] - nodes_data['pos'][src_ids]
    edges_data['dist'] = np.linalg.norm(edges_data['diff'], axis=-1, keepdims=True)
    edges_data['dir'] = unit_vector(edges_data['diff'])
    edges_data['temporal_mask'] = temporal_matrix[src_ids, dst_ids]
    edges_data['spatial_mask'] = 1 - edges_data['temporal_mask']

    for attr in EDGE_ATTRS:
        g.edata[attr] = torch.tensor(np.reshape(edges_data[attr], (g.num_edges(), STATE_DIMS[attr])), dtype=torch.float)

    return g
