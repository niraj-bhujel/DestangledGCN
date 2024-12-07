import math
import random
import torch
import numpy as np

from data.preprocess_utils import motion_kinematics

angles = np.arange(0, 360)

def rotate_pos(x, alpha):
    M = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    return np.dot(x, M)


def augment_graph_data(g, probability=0.5, dt=0.4, obsv_len=8):
    
    if random.uniform(0, 1) > probability:
        return g

    alpha = random.choice(angles) * math.pi / 180 # radians

    for tid in g.ndata['tid'].unique():
        
        tid_idx = g.ndata['tid']==tid
        tid_ntx = g.ndata['ntx'][tid_idx]
        
        p = g.ndata['pos'][tid_idx].numpy()
        p = rotate_pos(p, alpha)
        
        p, r, v, v_norm, v_dir, a, a_norm, h, goal, speed = motion_kinematics(p, dt, obsv_idx=torch.where(tid_ntx==obsv_len-1)[0].item()) 

        g.ndata['pos'][tid_idx] = torch.tensor(p, dtype=torch.float)
        g.ndata['rel'][tid_idx] = torch.tensor(r, dtype=torch.float)
        
        g.ndata['vel'][tid_idx] = torch.tensor(v, dtype=torch.float)
        g.ndata['vnorm'][tid_idx] = torch.tensor(v_norm, dtype=torch.float)

        g.ndata['acc'][tid_idx] = torch.tensor(a, dtype=torch.float)
        g.ndata['anorm'][tid_idx] = torch.tensor(a_norm, dtype=torch.float)

        g.ndata['dir'][tid_idx] = torch.tensor(v_dir, dtype=torch.float)

        g.ndata['hed'][tid_idx] = torch.tensor(h, dtype=torch.float)
        g.ndata['goal'][tid_idx] = torch.tensor(goal, dtype=torch.float)
    
    return g
    
if __name__ == '__main__':
    import time
    from copy import deepcopy
    from vis_graph import network_draw

    random.seed(42)
    start = time.time()
    g = seq_graphs

    # rotate pos
    ag = augment_graph_data(deepcopy(g), probability=1.0, angle = random.choice(range(0, 360)))
    ag.ndata['rel'] = ag.ndata['pos'] - get_nodes_pos_at_t(ag, t=7)
    
    print('Time required:', time.time() - start)
    # # verify all nodes are rotated
    # network_draw(g, fig_name='orig_graph', show_fig=True)
    # network_draw(ag, fig_name='augmented_graph', show_fig=True)
    
    # verify rel, vel, acc are changed/unchanged with rotation
    old_ped_pos = [g.ndata['pos'][g.ndata['tid']==tid].cpu().numpy() for tid in g.ndata['tid'].unique()]
    old_ped_rel = [g.ndata['rel'][g.ndata['tid']==tid].cpu().numpy() for tid in g.ndata['tid'].unique()]
    old_ped_vel = [g.ndata['vel'][g.ndata['tid']==tid].cpu().numpy() for tid in g.ndata['tid'].unique()]
    old_ped_acc = [g.ndata['acc'][g.ndata['tid']==tid].cpu().numpy() for tid in g.ndata['tid'].unique()]
    
    aug_ped_pos = [ag.ndata['pos'][ag.ndata['tid']==tid].cpu().numpy() for tid in ag.ndata['tid'].unique()]
    aug_ped_rel = [ag.ndata['rel'][ag.ndata['tid']==tid].cpu().numpy() for tid in ag.ndata['tid'].unique()]
    aug_ped_vel = [ag.ndata['vel'][ag.ndata['tid']==tid].cpu().numpy() for tid in ag.ndata['tid'].unique()]
    aug_ped_acc = [ag.ndata['acc'][ag.ndata['tid']==tid].cpu().numpy() for tid in ag.ndata['tid'].unique()]
    
    
    old_ped_p = old_ped_pos[0]
    old_ped_r = old_ped_rel[0]
    old_ped_v = old_ped_vel[0]
    old_ped_a = old_ped_acc[0]
    
    aug_ped_p = aug_ped_pos[0]
    aug_ped_r = aug_ped_rel[0]
    aug_ped_v = aug_ped_vel[0]
    aug_ped_a = aug_ped_acc[0]
    
    print('pos', old_ped_p == aug_ped_p)
    print('rel', old_ped_r == aug_ped_r)
    print('vel', old_ped_v == aug_ped_v) # augmented velocity is not same as old velocity
    print('acc', old_ped_a == aug_ped_a)