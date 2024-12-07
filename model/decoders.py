import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from data.states import STATE_DIMS, INTERACTION_RADIUS
from nets.gcn_net import *
from layers.mlp_layer import *
from model.losses import loss_func_param
from utils.train_utils import logits_to_dict

class MLPDec(nn.Module):
    def __init__(self, net_params, **kwargs):
        super().__init__()
        self.type = net_params['type']
        self.input_dim = net_params['in_dim']
        self.output_dim = net_params['out_dim']
        self.hidden_size = net_params['hidden_size']
        self.activation = net_params['activation']
        
        self.mlp = MLP(self.input_dim, self.output_dim, self.hidden_size, activation=net_params['activation'])

    def forward(self, z, x, x_0, num_samples=1, device='cuda:0'):
        zx = torch.cat([z, x, x_0], dim=-1)

        return self.mlp(zx)


class RNNDec(nn.Module):
    def __init__(self, net_params, **kwargs):
        super().__init__()
        self.type = net_params['type']
        self.input_dim = net_params['in_dim']
        self.hidden_dim = net_params['hidden_dim']
        self.output_dim = net_params['out_dim']

        self.pred_len = net_params['pred_seq_len']
        self.loss_func = net_params['traj_loss']

        input_state_dims = sum(STATE_DIMS[state] for state in kwargs['node_inputs'])
        pred_state_dims = sum(STATE_DIMS[state] for state in kwargs['node_outputs'])
        
        self.inital_action = nn.Linear(input_state_dims, pred_state_dims)

        self.initial_h = nn.Linear(self.input_dim-pred_state_dims, self.hidden_dim)

        self.rnn_cell = nn.GRUCell(self.input_dim, self.hidden_dim)
        
        self.rnn_output = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, z, x, x_0, num_samples=1, device='cuda:0'):
        
        zx = torch.cat([z, x], dim=-1)

        initial_state = self.initial_h(zx)
        # initial_state = torch.zeros((zx.shape[0], self.hidden_dim), dtype=torch.double).to(z.device)

        # infer initial action from initial states
        x_0 = self.inital_action(x_0)

        state = initial_state
        input_ = torch.cat([zx, x_0], dim=-1)

        rnn_outputs = []
        for j in range(self.pred_len):
            
            h_state = self.rnn_cell(input_, state)
            
            output = self.rnn_output(h_state)

            x_t = sample_traj(output, self.loss_func)[0]

            state = h_state

            input_ = torch.cat([zx, x_t] , dim=-1)

            rnn_outputs.append(output.view(num_samples, -1, self.output_dim))

        # (pred_len, num_samples, N, dim) -> (num_samples, N, pred_len, dim)
        rnn_outputs = torch.stack(rnn_outputs, dim=0).permute(1, 2, 0, 3) 
        return rnn_outputs

class GoalDec(nn.Module):

    def __init__(self, net_params, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.net_params = net_params

        self.goal_inputs = net_params['goal_inputs']

        # self.hidden_dim = net_params['net']['hidden_dim']
        # self.num_layer = net_params['net']['num_layers']

        self.obsv_len = net_params['obsv_seq_len']
        self.pred_len = net_params['pred_seq_len']

        input_dim = sum(STATE_DIMS[s] for s in net_params['goal_inputs']) + 1
        output_dim = loss_func_param(net_params['goal_loss'], 2)

        # self.inp = nn.Linear(input_dim, net_params['net']['hidden_dim'])
        # Layer = eval(net_params['net']['layer'])
        # self.net = nn.ModuleList([Layer(net_params['net']['hidden_dim'], 
        #                                 num_heads=net_params['net']['num_heads'],
        #                                 attn_dropout=net_params['net']['attn_dropout'],
        #                                 ) for _ in range(net_params['net']['num_layers'])])
        # self.out = nn.Linear(net_params['net']['hidden_dim'], output_dim)

    def forward(self, g, x, e=None, comm_traj=None, num_samples=1):


        t_pred = torch.stack([ntx[-1]-self.obsv_len for ntx in get_ndata(g, 'ntx', comm_traj)]).unsqueeze(1) #[traj, 1]

        x = torch.cat([x, t_pred], dim=-1)

        x = self.inp(x)

        for layer in self.net:
            x = layer(x)
        
        x = self.out(x)

        return x

class _STDec(nn.Module):
    '''
    This module is for testing the spatial deocder capability only. To use this moduel, set st_dec to true in arguments
    python train.py --version semi  --gpu_id 0 --model STDec --node_types VEHICLE --test_bs 64 --train_bs 64 --include_lane --debug --single_batch --st_dec --K_eval 1
    st-dec work best with l2_loss.
    '''
    def __init__(self, net_params, **kwargs):
        super().__init__()

        # adjust common dims

        input_node_dim = sum(STATE_DIMS[s] for s in net_params['node_inputs'])
        output_node_dim = sum(STATE_DIMS[s] for s in net_params['node_outputs']) 

        input_edge_dim = sum(STATE_DIMS[s] for s in net_params['edge_inputs']) 
        output_edge_dim = sum(STATE_DIMS[s] for s in net_params['edge_outputs'])


        # adjust dims for st_dec
        net_params['net']['in_dim_node'] = input_node_dim + (2 if net_params['sample_goal']>0 else 0)
        net_params['net']['in_dim_edge'] = input_edge_dim
        net_params['net']['out_dim_node'] = loss_func_param(net_params['node_loss'], output_node_dim)
        net_params['net']['out_dim_edge'] = loss_func_param(net_params['node_loss'], output_edge_dim)
        net_params['net']['mlp_readout_node'] = True if net_params['node_loss_wt']>0 else False
        net_params['net']['mlp_readout_edge'] = True if net_params['edge_loss_wt']>0 else False
        
        net_params['net']['hidden_dim'] = net_params['hidden_dim']
        net_params['net']['num_layers'] = net_params['num_layers']

        self.st_dec = eval(net_params['net']['type'])(net_params['net'], **kwargs)
        self.sample_goal = net_params['sample_goal']>0


    def forward(self, g, y, e, comm_traj=None, num_samples=1, device=torch.device('cpu'), **kwargs):
        '''
        y: tensor of shape (num_samples, num_nodes, ndim)
        e: tensor of shape (num_samples, num_nodes, ndim)
        '''

        # process multiple samples at once
        
        # y = y.permute(1, 0, 2)
        # e = e.permute(1, 0, 2)
            
        # _, y, e = self.st_dec(g, y, e)
        
        # y = y.permute(1, 0, 2)
        # e = e.permute(1, 0, 2)

        # return y, e

        # process per sample to prevent CUDA Out of Memory, specially during evaluation where K_eval=10
        pred_y = []
        pred_e = []
        with g.local_scope():

            if self.sample_goal:
                goals = tdist.normal.Normal(g.ndata['goal'], scale=1.).rsample((num_samples, ))
                y = torch.cat([y, goals], dim=-1)

            for i in range(num_samples):
                
                _, _y, _e = self.st_dec(g, y[i], e[i])

                pred_y.append(_y)
                pred_e.append(_e)
            
            pred_y = torch.stack(pred_y, dim=0)
            pred_e = torch.stack(pred_e, dim=0)
        
            return pred_y, pred_e

class STDec(nn.Module):
    '''
    This module is for testing the spatial deocder capability only. To use this moduel, set st_dec to true in arguments
    python train.py --version semi  --gpu_id 0 --model STDec --node_types VEHICLE --test_bs 64 --train_bs 64 --include_lane --debug --single_batch --st_dec --K_eval 1
    st-dec work best with l2_loss.
    '''
    def __init__(self, net_params, **kwargs):
        super().__init__()

        self.sample_goal = net_params['sample_goal']>0

        self.input_nodes = net_params['node_inputs']
        self.input_edges = net_params['edge_inputs']

        # adjust dims for st_dec
        net_params['net']['in_dim_node'] = {s:STATE_DIMS[s] for s in net_params['node_inputs']}
        net_params['net']['in_dim_edge'] = {s:STATE_DIMS[s] for s in net_params['edge_inputs']}
        net_params['net']['out_dim_node'] = {s:loss_func_param(net_params['node_loss'][s], STATE_DIMS[s]) for s in net_params['node_outputs']}
        net_params['net']['out_dim_edge'] = {s:loss_func_param(net_params['edge_loss'][s], STATE_DIMS[s]) for s in net_params['edge_outputs']}
        net_params['net']['mlp_readout_node'] = True if len(net_params['node_outputs'])>0 else False
        net_params['net']['mlp_readout_edge'] = True if len(net_params['edge_outputs'])>0 else False
        
        net_params['net']['hidden_dim'] = net_params['hidden_dim']
        net_params['net']['num_layers'] = net_params['num_layers']

        self.st_dec = eval(net_params['net']['type'])(net_params['net'], dt=net_params['dt'])


    def forward(self, g, y, e, comm_traj=None, num_samples=1, device=torch.device('cpu'), **kwargs):
        '''
        y: tensor of shape (num_samples, num_nodes, ndim)
        e: tensor of shape (num_samples, num_nodes, ndim)
        '''

        # # process multiple samples at once
        
        # y = {s:v.permute(1, 0, 2) for s, v in y.items()}
        # e = {s:v.permute(1, 0, 2) for s, v in e.items()}
            
        # _, y, e = self.st_dec(g, y, e)
        
        # y = {s:v.permute(1, 0, 2) for s, v in y.items()}
        # e = {s:v.permute(1, 0, 2) for s, v in e.items()}

        # return y, e

        # process per sample to prevent CUDA Out of Memory, specially during evaluation where K_eval=10
        pred_y = defaultdict(list)
        pred_e = defaultdict(list)

        for i in range(num_samples):
            
            yi = {s:val[i] for s, val in y.items()}
            ei = {s:val[i] for s, val in e.items()}

            with g.local_scope():
                _, yi, ei = self.st_dec(g, yi, ei)

            for s, v in yi.items():
                pred_y[s].append(v)

            for s, v in ei.items():
                pred_e[s].append(v)
        
        pred_y = {s:torch.stack(v, dim=0) for s, v in pred_y.items()}
        pred_e = {s:torch.stack(v, dim=0) for s, v in pred_e.items()}
    
        return pred_y, pred_e