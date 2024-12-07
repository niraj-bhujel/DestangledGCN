# nuScenes dev-kit.
# Code written by Freddy Boulton, Robert Beaudoin 2020.
import abc
import math
from typing import Tuple
import dgl
import torch
from copy import copy, deepcopy
KinematicsData = Tuple[float, float, float, float, float, float, float, float, float, float]

def angle_diff(x, y, period):
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > math.pi:
        diff = diff - (2 * torch.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def _kinematics(g: dgl.DGLHeteroGraph, tid: int) -> KinematicsData:
    
    traj_idxs = g.ndata['tid']==tid
    
    x, y = g.ndata['pos'][traj_idxs][-1] # (2, )
    vx, vy = g.ndata['vel'][traj_idxs][-1]
    ax, ay = g.ndata['acc'][traj_idxs][-1]

    velocity = g.ndata['vnorm'][traj_idxs][-1]
    acceleration = g.ndata['anorm'][traj_idxs][-1]

    current_yaw, = g.ndata['yaw'][traj_idxs][-1]
    prev_yaw, = g.ndata['yaw'][traj_idxs][-2]
    
    yaw_rate = angle_diff(current_yaw, prev_yaw, period=2*math.pi) / 0.5

    if torch.isnan(yaw_rate):
        yaw_rate = 0.0

    # hx, hy = torch.cos(current_yaw), torch.sin(current_yaw)
    # vx, vy = velocity * hx, velocity * hy
    # ax, ay = acceleration * hx, acceleration * hy

    return x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, current_yaw


def _constant_velocity_and_heading(kinematics_data: KinematicsData,
                                               sec_from_now: float,
                                               sampled_at: int) -> torch.Tensor:
    """
    Computes a constant velocity baseline for given kinematics data, time window
    and frequency.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, _, _, _, _ = kinematics_data
    
    preds = []
    time_step = 1.0 / sampled_at
    
    for time in torch.arange(time_step, sec_from_now + time_step, time_step):
        preds.append(torch.tensor([x + time * vx, y + time * vy]))
        
    return torch.stack(preds)


def _constant_acceleration_and_heading(kinematics_data: KinematicsData,
                                       sec_from_now: float, sampled_at: int) -> torch.Tensor:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the acceleration and heading are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, ax, ay, _, _, _, _ = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    for time in torch.arange(time_step, sec_from_now + time_step, time_step):
        half_time_squared = 0.5 * time * time
        preds.append(torch.tensor([x + time * vx + half_time_squared * ax,
                      y + time * vy + half_time_squared * ay]))
    return torch.stack(preds)


def _constant_speed_and_yaw_rate(kinematics_data: KinematicsData,
                                 sec_from_now: float, sampled_at: int) -> torch.Tensor:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the (scalar) speed and yaw rate are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, speed, yaw_rate, _, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    distance_step = time_step * speed
    yaw_step = time_step * yaw_rate
    
    x = deepcopy(x)
    y = deepcopy(y)
    
    for _ in torch.arange(time_step, sec_from_now + time_step, time_step):
        x += distance_step * torch.cos(yaw)
        y += distance_step * torch.sin(yaw)
        preds.append(torch.tensor([x, y]))
        yaw += yaw_step
    return torch.stack(preds)


def _constant_magnitude_accel_and_yaw_rate(kinematics_data: KinematicsData,
                                           sec_from_now: float, sampled_at: int) -> torch.Tensor:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the rates of change of speed and yaw are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, speed, yaw_rate, accel, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    speed_step = time_step * accel
    yaw_step = time_step * yaw_rate

    x = deepcopy(x)
    y = deepcopy(y)
    
    for _ in torch.arange(time_step, sec_from_now + time_step, time_step):
        distance_step = time_step * speed
        x += distance_step * torch.cos(yaw)
        y += distance_step * torch.sin(yaw)
        preds.append(torch.tensor([x, y]))
        speed += speed_step
        yaw += yaw_step
        
    return torch.stack(preds)

class Baseline(abc.ABC):

    def __init__(self, sec_from_now=6., sampled_at=2,):
        """
        Inits Baseline.
        :param sec_from_now: How many seconds into the future to make the prediction.
        :param helper: Instance of PredictHelper.
        """
        #assert sec_from_now % 0.5 == 0, f"Parameter sec_from_now must be divisible by 0.5. Received {sec_from_now}."
        self.sec_from_now = sec_from_now
        self.sampled_at = sampled_at  # 2 Hz between annotations.
        self.name = 'Physics'

    @abc.abstractmethod
    def __call__(self, token: str):
        pass

    def eval(self, ):
        self.training=False

    def __repr__(self):
        return f"name: {self.name}, sec_from_now: {self.sec_from_now}, sampled_at:{self.sampled_at}"


class ConstantVelocity(Baseline):
    """ Makes predictions according to constant velocity and heading model. """

    def __call__(self, g, traj_ids=None, device='cpu', **kwargs):
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """

        if traj_ids is None:
            traj_ids = g.ndata['tid'].unique()
        
        preds = [_constant_velocity_and_heading(_kinematics(g, tid), self.sec_from_now, self.sampled_at) 
                    for tid in traj_ids]
            
        return torch.stack(preds).unsqueeze(0).to(device)
    
class ConstantAccleration(Baseline):
    """ Makes predictions according to constant velocity and heading model. """

    def __call__(self, g, traj_ids=None, **kwargs):
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """

        if traj_ids is None:
            traj_ids = g.ndata['tid'].unique()
        
        preds = [_constant_acceleration_and_heading(_kinematics(g, tid), self.sec_from_now, self.sampled_at) 
                    for tid in traj_ids]
            
        return preds


class ConstantSpeed(Baseline):
    """ Makes predictions according to constant velocity and heading model. """

    def __call__(self, g, traj_ids=None, **kwargs):
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """

        if traj_ids is None:
            traj_ids = g.ndata['tid'].unique()
        
        preds = [_constant_speed_and_yaw_rate(_kinematics(g, tid), self.sec_from_now, self.sampled_at) 
                    for tid in traj_ids]
            
        return preds

class ConstantMagnitude(Baseline):
    """ Makes predictions according to constant velocity and heading model. """

    def __call__(self, g, traj_ids=None, **kwargs):
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """

        if traj_ids is None:
            traj_ids = g.ndata['tid'].unique()
        
        preds = [_constant_magnitude_accel_and_yaw_rate(_kinematics(g, tid), self.sec_from_now, self.sampled_at) 
                    for tid in traj_ids]
            
        return preds


class PhysicsOracle(Baseline):
    """ Makes several physics-based predictions and picks the one closest to the ground truth. """

    def __call__(self, g, traj_ids=None, **kwargs):
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """

        if traj_ids is None:
            traj_ids = g.ndata['tid'].unique()

        path_funs = [
            _constant_velocity_and_heading,
            _constant_acceleration_and_heading,
            _constant_speed_and_yaw_rate,
            _constant_magnitude_accel_and_yaw_rate, # BUG! constant magnitude did inplace addition!!
        ]
        
        oracle_preds = []
        for tid in traj_ids:
            kinematics = _kinematics(g, tid)
            paths = torch.stack([path_fun(kinematics, self.sec_from_now, self.sampled_at) for path_fun in path_funs])
            
            oracle_preds.append(paths)
            
        return torch.stack(oracle_preds).permute(1, 0, 2, 3).to(g.device)
            
        
if __name__=='__main__':
    
    g = obsv_graphs
    gt = gt_graphs
    
    cv_preds = ConstantVelocity()(g, comm_traj)
    # ca_preds = ConstantAccleration()(g, comm_traj)
    # cs_preds = ConstantSpeed()(g, comm_traj)
    # cm_preds = ConstantMagnitude()(g, comm_traj)
    oracle_preds = PhysicsOracle()(g, comm_traj)
    
    # gt_pos = [gt.ndata['pos'][gt.ndata['tid']==tid] for tid in comm_traj]
    # cv_err = [torch.linalg.norm(pred[:len(gt)] - gt).mean() for pred, gt in zip(cv_preds, gt_pos)]
    # ca_err = [torch.linalg.norm(pred[:len(gt)] - gt).mean() for pred, gt in zip(ca_preds, gt_pos)]
    # cs_err = [torch.linalg.norm(pred[:len(gt)] - gt).mean() for pred, gt in zip(cs_preds, gt_pos)]
    # cm_err = [torch.linalg.norm(pred[:len(gt)] - gt).mean() for pred, gt in zip(cm_preds, gt_pos)]

    # oracle_err = [torch.linalg.norm(pred[:len(gt)] - gt).mean() for pred, gt in zip(oracle_preds, gt_pos)]
    
    
    # print('CV Err:', torch.stack(cv_err).mean())
    # print('CA Err:', torch.stack(ca_err).mean())
    # print('CS Err:', torch.stack(cs_err).mean())
    # print('CM Err:', torch.stack(cm_err).mean())

    # print('Oracle Error:', torch.stack(oracle_err).mean())
    
    
    
    
    