import time
import numpy as np
from scipy.spatial.distance import cdist
import torch
from collections import defaultdict

def rank_metric_over_top_k_modes(metric_results: np.ndarray,
                                 mode_probabilities: np.ndarray,
                                 ranking_func: str) -> np.ndarray:
    """
    Compute a metric over all trajectories ranked by probability of each trajectory.
    :param metric_results: 1-dimensional array of shape [batch_size, num_modes].
    :param mode_probabilities: 1-dimensional array of shape [batch_size, num_modes].
    :param ranking_func: Either 'min' or 'max'. How you want to metrics ranked over the top
            k modes.
    :return: Array of shape [num_modes].
    """

    if ranking_func == "min":
        func = np.minimum.accumulate
    elif ranking_func == "max":
        func = np.maximum.accumulate
    else:
        raise ValueError(f"Parameter ranking_func must be one of min or max. Received {ranking_func}")

    p_sorted = np.flip(mode_probabilities.argsort(axis=-1), axis=-1)
    indices = np.indices(metric_results.shape)

    sorted_metrics = metric_results[indices[0], p_sorted]

    return func(sorted_metrics, axis=-1)

def rank_predictions_over_top_k_modes(prediction, probs, top_k=10):
    '''
    prediction: (K, N, T, 2)
    probs : (K, N)
    '''
    num_modes, batch_size, _, _ = prediction.shape
    
    min_k =  min(num_modes, top_k)

    topk_probs, indices = torch.topk(probs, min_k, dim=0)
    
    batch_indices = torch.arange(batch_size).unsqueeze(0).repeat(min_k, 1)
    
    return prediction[indices, batch_indices], topk_probs
    
def compute_ade(predicted_traj, gt_traj, pred_len=12):
    predicted_traj = predicted_traj[:, :pred_len, :]
    gt_traj = gt_traj[:, :pred_len, :]

    error = np.linalg.norm(predicted_traj - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade.flatten()


def compute_fde(predicted_traj, gt_traj, pred_len=12):
    predicted_traj = predicted_traj[:, :pred_len, :]
    gt_traj = gt_traj[:, :pred_len, :]
    
    final_error = np.linalg.norm(predicted_traj[:, -1, :] - gt_traj[:, -1, :], axis=-1)
    return final_error.flatten()

def compute_ade_fast(y_true, y_pred, masks):

    '''
    y_true: (N, T, 2)
    y_pred: (K, N, T, 2)
    masks: (N, T) - 1 indicate no gt true
    '''
    
    if torch.any(y_true.isnan()):
        y_true[y_true.isnan()] = 0.

    num_modes = y_pred.shape[0]

    y_true = y_true.repeat(num_modes, 1, 1, 1) # (K, N, T, 2)
    masks = masks.unsqueeze(0).repeat(num_modes, 1, 1) # (K, N, T)
    assert y_true.shape == y_pred.shape, f"y_true shape is {y_true.shape} and y_pred shape is {y_pred.shape}"
    
    err =  y_true - y_pred
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=-1)
    err = torch.pow(err, exponent=0.5) # (K, N, T)

    err = torch.sum(err * (1-masks), dim=2)/torch.sum(1-masks, dim=2) # (K, N)

    return err

def compute_fde_fast(y_true, y_pred, masks):
    '''
    y_true: (N, T, 2)
    y_pred: (K, N, T, 2)
    masks: (N, T) - 1 indicate no gt true
    '''
    
    if torch.any(y_true.isnan()):
        y_true[y_true.isnan()] = 0.

    num_modes= y_pred.shape[0]
        
    gt_len = torch.sum(1-masks, dim=1) # (N, )
    
    indices = gt_len[None, :, None, None].repeat(num_modes, 1, 1, 2).type(torch.int64) - 1
    
    y_true = y_true.repeat(num_modes, 1, 1, 1)
    y_true = torch.gather(y_true, dim=2, index=indices)
    y_pred = torch.gather(y_pred, dim=2, index=indices)
    
    assert y_true.shape == y_pred.shape, f"y_true shape is {y_true.shape} and y_pred shape is {y_pred.shape}"
    
    err =  y_true - y_pred
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=-1)
    err = torch.pow(err, exponent=0.5)# (K, N)

    return err.squeeze(-1)

def compute_ade_fde_k(gt_traj, pred_traj, probs, masks, top_k=0):
    '''
    gt_traj: tensor of shape (N, T, 2) 
    pred_traj: tensor of shape [K, N, T, 2]
    probs: tensor of shape (K, N)
    masks: tensor of shape (N, T)
    '''

    if top_k>0 and probs is not None:
        pred_traj, _ = rank_predictions_over_top_k_modes(pred_traj, probs, top_k=top_k)

    ade = compute_ade_fast(gt_traj, pred_traj, masks)
    
    fde = compute_fde_fast(gt_traj, pred_traj, masks)
    
    return ade, fde

def compute_min_ade(y_true, y_pred, masks):
    '''
    y_true: (N, T, 2)
    y_pred: (K, N, T, 2)
    masks: (N, T) - 1 indicate no gt true
    '''
    
    err = compute_ade_fast(y_true, y_pred, masks) 
    
    min_err, min_indices = torch.min(err, dim=0)
    
    return min_err, min_indices
    
def compute_ade_fde(gt_pos, pred_pos):
    '''
    gt_pos: list of N tensor each with shape [T, 2], T may be different for each N 
    pred_pos: list of N tensor each of shape [K, T, 2], could be array of shape [N, K, T, 2]
    '''
    ade = []
    fde = []

    if isinstance(gt_pos[0], torch.Tensor):
        gt_pos = [pos.cpu().numpy() for pos in gt_pos]

    if isinstance(pred_pos[0], torch.Tensor):
        pred_pos = pred_pos.cpu().numpy()
        
    # compute ade, fde for all traj (filter later)
    for i in range(len(gt_pos)):

        gt_traj = gt_pos[i][None, :, :]
        pred_traj = pred_pos[i]
        
        ade.append(compute_ade(pred_traj, gt_traj, pred_len=len(gt_pos[i])))
        fde.append(compute_fde(pred_traj, gt_traj, pred_len=len(gt_pos[i])))

    return ade, fde

def compute_ade_fde_per_steps(gt_pos, pred_pos, probs, gt_masks, top_k=25, pred_lengths = [2, 4, 6, 8, 10, 12]):

    num_modes, batch_size, _, _ = pred_pos.shape

    _, mode_indices = probs[:top_k].max(0)
    
    batch_indices = torch.arange(batch_size)

    ret = defaultdict(dict)
    

    for t in reversed(sorted(pred_lengths)):
                
        ade = compute_ade_fast(gt_pos[..., :t, :], pred_pos[..., :t, :], gt_masks[..., :t])
        fde = compute_fde_fast(gt_pos[..., :t, :], pred_pos[..., :t, :], gt_masks[..., :t])
        

        ret[t]['ade_full'] = ade.mean().item()
        ret[t]['fde_full'] = fde.mean().item()
        ret[t]['ade_ml'] = ade[mode_indices, batch_indices].mean().item()
        ret[t]['fde_ml'] = fde[mode_indices, batch_indices].mean().item()
        ret[t]['ade_best'] = ade[:top_k].min(0).values.mean().item()
        ret[t]['fde_best'] = fde[:top_k].min(0).values.mean().item()
        
    return ret
    

def pairwise_euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
    
def final_l2(path1, path2):
    '''
    Parameters
    ----------
    path1 : array of [T, 2] 
    path2 : Tarray of [T, 2] 
    n_predictions : prediction length. 
        DESCRIPTION. The default is 12.
    '''
    
    row1 = path1[-1]
    row2 = path2[-1]
    return np.linalg.norm((row2[0] - row1[0], row2[1] - row1[1]))

def average_l2(path1, path2, n_predictions=12):
    '''
    Parameters
    ----------
    path1 : array of [T, 2] 
    path2 : Tarray of [T, 2] 
    n_predictions : prediction length. 
        DESCRIPTION. The default is 12.
    '''
    assert len(path1) >= n_predictions
    assert len(path2) >= n_predictions
    
    path1 = path1[:n_predictions]
    path2 = path2[:n_predictions]

    return sum(np.linalg.norm((r1[0] - r2[0], r1[1] - r2[1]))
               for r1, r2 in zip(path1, path2)) / n_predictions



def collision(path1, path2, frames1=None, frames2=None, n_predictions=12, person_radius=0.1, inter_parts=2):
    """Check if there is collision or not
    Parameters
    ----------
    path1 : array of [T, 2] 
    path2 : Tarray of [T, 2]     
    """

    assert len(path1) >= n_predictions
    
    # path1 = path1[-n_predictions:]
    # frames1 = set(frames1)
    # frames2 = set(frames2)
    # common_frames = frames1.intersection(frames2)
    
    # # If there is no interaction, there is no collision
    # if not common_frames:
    #     return False
    
    # path1 = [path1[i] for i in range(len(path1)) if path1[i].frame in common_frames]
    # path2 = [path2[i] for i in range(len(path2)) if path2[i].frame in common_frames]
    
    if frames1 is not None:
        if frames2 is not None:
            common_frames = np.intersect1d(frames1, frames2)
        
            if common_frames.size==0:
                return False

            path1 = np.array([path1[i] for i in range(len(path1)) if frames1[i] in common_frames])
            path2 = np.array([path2[i] for i in range(len(path2)) if frames2[i] in common_frames])
    
    def getinsidepoints(p1, p2, parts=2):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1))) #[2, parts+1]

    for i in range(len(path1) - 1):
        p1, p2 = [path1[i][0], path1[i][1]], [path1[i + 1][0], path1[i + 1][1]] #[(x1, y1), (x2, y2)]
        p3, p4 = [path2[i][0], path2[i][1]], [path2[i + 1][0], path2[i + 1][1]] #[(x1, y1), (x2, y2)]
        
        inside_points1 = getinsidepoints(p1, p2, inter_parts)
        inside_points2 = getinsidepoints(p3, p4, inter_parts)
        
        inter_parts_dist = np.linalg.norm(inside_points1 - inside_points2, axis=0)
        
        if np.min(inter_parts_dist) <= 2 * person_radius:
            return True
        
    return False


#%%
if __name__=='__main__':
    import time
    # num_samples = 100
    # y_pred = np.random.randint(-15, 15, size=(12, num_samples, 2)) + np.random.rand(12, num_samples, 2)
    # y_ids = np.random.randint(1, y_pred.shape[1]/2, (y_pred.shape[1], ))
    
    # y_pred = target_p.permute(2, 0, 1).cpu().numpy() #channel first
    y_pred = target_p.permute(1, 0, 2).cpu().numpy() #channel last
    y_ids = ped_id.cpu().numpy().flatten()
    y_frames = target_f.cpu().numpy()
    
    ########## using default #########################
    start = time.time()
    num_peds = y_pred.shape[1]
    collision_matrix = np.zeros((num_peds, num_peds))
    for i in range(num_peds):
       for j in range(i+1, num_peds):
            if y_ids[i]!=y_ids[j]:
                path1, frames1 = y_pred[:, i, :], y_frames[i]
                path2, frames2 = y_pred[:, j, :], y_frames[j]
                if collision(path1, path2, frames1, frames2, inter_parts=1):
                    collision_matrix[i, j]=True
                    collision_matrix[j, i]=True
    collision_rewards = ~np.any(collision_matrix, axis=1, keepdims=True)
    print('%f secs, samples: %d, collisions: %d, rewards: %f'%(time.time()-start, num_peds, 
                                                               np.count_nonzero(collision_matrix)/2, 
                                                               np.mean(collision_rewards)))
    
    #%%
    ############ Using torch ############
    import torch
    # y_pred = pred_vel
    y_pred = target_p
    # y_pred = target_p.permute(2, 0, 1) # if channel first
    # y_pred = target_p.permute(1, 0, 2) # if channel last
    # y_pred = organize_tensor(logits_v, args.data_format, 2, reshape=True, time_major=True)
    y_ids = ped_id.cpu().numpy().flatten()
    y_frames = target_f.cpu().numpy()
    
    # #test on few samples only
    # y_pred = y_pred[:, :5, :]
    # y_ids = y_ids[:5]
    # y_frames= y_frames[:5, :]
    # num_peds = y_pred.shape[1]
    
    start = time.time()
    collision_peds = []
    non_collision_peds = []
    for t in range(12):
        paths = y_pred[t, :, :]        
        peds_dist = pairwise_euclidean_dist(paths, paths)
        collision_peds_t = (peds_dist<=0.2).type_as(peds_dist) * peds_dist # >0 means collision, otherwise non-collision
        # collision_peds_t_numpy = collision_peds_t.cpu().numpy()
        collision_peds_t[collision_peds_t>1e-3]=1
        collision_peds.append(collision_peds_t)
        
        # non_collision_peds_t = (peds_dist>2).type_as(peds_dist) * peds_dist # >0 mean no collision, otherwise collisions        
        # non_collision_peds.append(non_collision_peds_t)
    
    collision_peds = torch.stack(collision_peds, dim=0).sum(dim=0)
    #NOTE: By default same pedestrian will have collisions, as their distance is less than 0.2
    # Also multiple samples/row may have same ids, so need to flag of these pedestrian from the collision_matrix, 
    # Once the mask is computed, invert it to flag off those similar pedestrian and
    #finally multiply it with collision matirx that will set the flag to False for these similar pedestrian. 
    peds_mask = np.array([id_ == y_ids for id_ in y_ids]) #1-same peds, 0-different peds
    peds_mask =  torch.from_numpy(~peds_mask).type_as(collision_peds)
    collision_peds *= peds_mask#same peds will not have collision
    # non_collision_peds  += torch.from_numpy(~peds_mask).type_as(non_collision_peds) 
    
    #NOTE! Collision cannot occur in different frames. ONLY frame at same time step is considered. 
    # True value indicate pedestrian MAY have collisions
    frames_mask = np.array([np.any(f == y_frames, axis=1) for f in y_frames]) #1-same frames, 0-different frames
    collision_peds *= torch.from_numpy(frames_mask).type_as(collision_peds) #only same frames will have collision i.e 1
    # non_collision_peds += torch.from_numpy(~frames_mask).type_as(non_collision_peds)
    
    collision_peds_numpy = collision_peds.detach().cpu().numpy()
    num_collisions = torch.sum(collision_peds).type(torch.float)*0.5
    

    collision_rewards = collision_peds.sum(dim=1, keepdim=True)
    for i in range(collision_rewards.numel()):
        
        if collision_rewards[i]>0:
            collision_rewards[i]=-1
        else:
            collision_rewards[i]=1
                
    print('%f secs, samples: %d, collisions: %d, rewards:%f'%(time.time()-start, y_pred.shape[1], 
                                                               num_collisions, 
                                                               collision_rewards.mean()
                                                               ))
    #%%
    ############################# USING scipy.distance cdist (FASTER than Numpy) #####################
    # Time required ~ 0.02262secs
    # y_pred = target_p.permute(2, 0, 1).cpu().numpy() # if channel first
    y_pred = target_p.permute(1, 0, 2).cpu().numpy() # if channel last
    y_ids = ped_id.cpu().numpy().flatten()
    y_frames = target_f.cpu().numpy()
    
    start = time.time()
    num_peds = y_pred.shape[1]
    
    collision_matrix = np.zeros((num_peds, num_peds))
    
    for t in range(12):
        paths = y_pred[t, :, :]
        collistions_t = cdist(paths, paths)<=0.2 #collision if dist<=person_radius*2
        collision_matrix = np.logical_or(collision_matrix, collistions_t)
    
    #NOTE: By default same pedestrian will have collisions, as their distance is less than 0.2
    # Also multiple samples/row may have same ids, so need to flag of these pedestrian from the collision_matrix, 
    # Once the mask is computed, invert it to flag off those similar pedestrian and
    #finally multiply it with collision matirx that will set the flag to False for these similar pedestrian. 
    peds_mask = ~np.array([id_ == y_ids for id_ in y_ids], dtype=bool)
    collision_matrix  = np.logical_and(collision_matrix, peds_mask)
    
    #NOTE! Collision cannot occur in different frames. ONLY frame at same time step is considered. 
    # True value indicate pedestrian MAY have collisions
    frames_mask = np.array([np.any(f == y_frames, axis=1) for f in y_frames], dtype=bool) 
    collision_matrix = np.logical_and(collision_matrix, frames_mask)
    collision_rewards = ~np.any(collision_matrix, axis=1, keepdims=True) #reward=not(collision) for each ped                
    print('%f secs, samples: %d, collisions: %d, rewards: %f'%(time.time()-start, y_pred.shape[1], 
                                                               np.count_nonzero(collision_matrix)/2, 
                                                               np.mean(collision_rewards)))
    #%%
    ######################## using native numpy method #########################

    y_pred = target_p.permute(1, 0, 2).cpu().numpy() # if channel last
    y_ids = ped_id.cpu().numpy().flatten()
    y_frames = target_f.cpu().numpy()
    
    start = time.time()
    collision_matrix = np.zeros((y_pred.shape[1], y_pred.shape[1]))
    
    for i in range(y_pred.shape[1]):

        for j in range(i+1, y_pred.shape[1]):
            if y_ids[i]!=y_ids[j]:
                path1 = y_pred[:, i, :]
                path2 = y_pred[:, j, :]
                #method 5
                # dist = np.diag(cdist(path1, path2))
                #method 4 -> 4.10secs
                # dist = [math.hypot(a[0]-b[0], a[1]-b[1]) for a, b in zip(path1, path2)]
                #method 3 -> 16.86secs
                # dist = [(a - b)**2 for a, b in zip(path1, path2)]
                # d = [math.sqrt(sum(d)) for d in dist]
                #method 2 -> 3.23secs
                dist = np.sqrt((path1[:, 0]-path2[:, 0])**2 + (path1[:, 1]-path2[:, 1])**2)
                #method 1 ->4.69secs
                # dist = scipy.linalg.norm(path1-path2, axis=1, keepdims=True)
                #method 0 (scipy.spatial.distance.eulidean)
                # dist = [euclidean(path1[i], path2[i]) for i in range(12)]
                if min(dist)<=0.2:
                    collision_matrix[i, j] = True
                    collision_matrix[j, i] = True
    # collision_rewards = ~np.any(collision_matrix, axis=1, keepdims=True) #reward=not(collision)
    print('%f secs, samples: %d, collisions: %d, rewards: %f'%(time.time()-start, y_pred.shape[1], 
                                                               np.count_nonzero(collision_matrix)/2, 
                                                               np.mean(collision_rewards)))
    
    #NOTE: Role of inter_parts (169 samples) 
    #parts = 0, collisions = 67, time = 0.23 secs
    #parts = 1, collisions = 67, time = 20.20 secs
    #parts = 2, collisions = 96, time = 20.90 secs
    
    #Using torch computation (169 samples) 
    #parts = 0, collision = 67, time = 2.20secs
    

    
