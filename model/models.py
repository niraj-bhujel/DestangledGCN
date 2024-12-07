#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:20:35 2020

@author: dl-asoro
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np

from data.states import STATE_DIMS, STATE_DIMS, INTERACTION_RADIUS
from layers.mlp_layer import MLP, MLPReadout
from nets.gcn_net import GCNNet
from nets.sa_gcn_net import SAGCNNet
from nets.san_net import SANNet
from model.decoders import MLPDec, RNNDec, STDec, GoalDec
from model.vae import VAE
from model.losses import *

from utils.graph_utils import *

class GCN(VAE):
    def __init__(self, net_params, **kwargs):
        super().__init__(net_params, **kwargs)
        self.name = 'GCN'
        self.kwargs = kwargs
        self.z_dim = net_params['z_dim']
        self.net = net_params['net']
        
        self.obsv_len = net_params['obsv_seq_len']
        self.pred_len = net_params['pred_seq_len']

        self.node_inputs = net_params['node_inputs']
        self.node_outputs = net_params['node_outputs']

        self.edge_inputs = net_params['edge_inputs']
        self.edge_outputs = net_params['edge_outputs']
        
        self.dt = net_params['dt']

        # self.traj_loss = net_params['traj_loss']
        self.cvae = nn.ModuleDict()
        
        self.sample_timesteps = False
        
        # input/output dimensions
        in_dim_node = {s:STATE_DIMS[s] for s in net_params['node_inputs']} 
        in_dim_edge = {s:STATE_DIMS[s] for s in net_params['edge_inputs']}

        out_dim_node =  {s:loss_func_param(net_params['node_loss'][s], STATE_DIMS[s]) for s in net_params['node_outputs']}
        out_dim_edge = {s:loss_func_param(net_params['edge_loss'][s], STATE_DIMS[s]) for s in net_params['edge_outputs']}

        #adjust param for past_enc
        net_params['past_enc']['in_dim_node'] = in_dim_node
        net_params['past_enc']['in_dim_edge'] = in_dim_edge
        net_params['past_enc']['out_dim_node'] = out_dim_node
        net_params['past_enc']['out_dim_edge'] = out_dim_edge

        #adjust param for target_enc
        net_params['target_enc']['in_dim_node'] = in_dim_node
        net_params['target_enc']['in_dim_edge'] = in_dim_edge
        net_params['target_enc']['out_dim_node'] = out_dim_node
        net_params['target_enc']['out_dim_edge'] = out_dim_edge

        net_params['past_enc'].update(net_params['net'])
        net_params['target_enc'].update(net_params['net'])

        self.cvae['past_enc'] = eval(net_params['net']['type'])(net_params['past_enc'])
        
        self.cvae['target_enc'] = eval(net_params['net']['type'])(net_params['target_enc'])

        # latent layers
        self.cvae['f_pz_x'] = MLP(net_params['enc_hdim'], net_params['z_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim']))
        self.cvae['f_qz_x_y'] = MLP(net_params['enc_hdim'], net_params['z_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim']))
        
        # traj_dec
        latent_dims = self.z_dim * (self.pred_len if self.sample_timesteps else 1)
 
        self.traj_dec = nn.ModuleDict()
        if net_params['traj_dec']['type']=='MLPDec':
            net_params['traj_dec']['in_dim'] = sum(in_dim_node.values()) + net_params['enc_hdim'] + latent_dims                
            for s in net_params['node_outputs']:
                net_params['traj_dec']['out_dim'] = net_params['pred_seq_len']*loss_func_param(net_params['node_loss'][s], STATE_DIMS[s])
                self.traj_dec[s] = eval(net_params['traj_dec']['type'])(net_params['traj_dec'], **kwargs)
                
        elif net_params['traj_dec']['type']=='RNNDec':
            net_params['traj_dec']['hidden_dim'] = net_params['dec_hdim']
            net_params['traj_dec']['in_dim'] = sum(out_dim_node.values()) + net_params['enc_hdim'] + latent_dims
            for s in net_params['node_outputs']:
                net_params['traj_dec']['out_dim'] = loss_func_param(net_params['node_loss'][s], STATE_DIMS[s])
                self.traj_dec[s] = eval(net_params['traj_dec']['type'])(net_params['traj_dec'], **kwargs)
        else:
            raise Exception(f"{net_params['net']['type']} not a valid dec type" )
            
            
        
    def forward(self, gx, xx, ex, gy=None, yy=None, ey=None, gt=None, comm_traj=None, num_samples=1, device=torch.device('cpu')):
        
        xx_cat = torch.cat([v for k,  v in xx.items()], dim=-1)
        #common tid in obsv and target graph
        if comm_traj is None:
            comm_traj = np.sort(np.intersect1d(gx.ndata['tid'].cpu().numpy(), gt.ndata['tid'].cpu().numpy()))
        
        curr_state_t = torch.stack([xx_cat[gx.ndata['tid']==tid][-1] for tid in comm_traj], dim=0) #[num_traj, ndim]

        #Encode X~p(X)
        gx, _, _ = self.cvae['past_enc'](gx, xx, ex)
        
        h_x = torch.stack([torch.mean(gx.ndata['h'][gx.ndata['tid']==tid], dim=0) for tid in comm_traj], dim=0) # [num_traj, hidden_dim]
        
        pz_x =  self.p_dist_from_h(self.cvae['f_pz_x'](h_x), self.z_dim, p_dist=self.pz_x) # p_distribution

        KLD = 0
        sample_size = (num_samples, self.pred_len) if self.sample_timesteps else (num_samples, )
        if self.training:
            #Encode X, Y ~ q(X, Y)
            gy, _, _ = self.cvae['target_enc'](gy, yy, ey)

            h_x_y = torch.stack([torch.mean(gy.ndata['h'][gy.ndata['tid']==tid], dim=0) for tid in comm_traj], dim=0)
            
            # h_x_y = torch.cat([h_x, h_y], dim=-1)
            qz_x_y = self.q_dist_from_h(self.cvae['f_qz_x_y'](h_x_y), self.z_dim, self.qz_x_y)
            
            z = qz_x_y.rsample(sample_size) #[num_samples, 12, num_traj, z_dim] or [num_samples, num_traj, z_dim]

            KLD = self.kl_divergence(qz_x_y, pz_x, samples=z)
            
        else:
            z = pz_x.sample(sample_size) #[num_samples, 12, num_traj, z_dim) or [num_samples, num_traj, z_dim)
        
        self.z = z

        if len(sample_size)==2:
            z = z.permute(0, 2, 1, 3).reshape(num_samples, len(comm_traj), self.pred_len*self.z_dim) # if z shape is (num_samples, 12, num_traj, z_dim)

        z = z.view(num_samples*len(comm_traj), -1)

        h_x = h_x.repeat(num_samples, 1, 1).view(num_samples*len(comm_traj), -1)
        h_x0 = curr_state_t.repeat(num_samples, 1, 1).view(num_samples*len(comm_traj), -1)

        y_zx = {s:self.traj_dec[s](z, h_x, h_x0, num_samples=num_samples, device=device) for s in self.node_outputs}
        
        logits_y = {s:y_zx[s].view(num_samples, len(comm_traj), self.pred_len, -1) for s in self.node_outputs} #[num_samples, num_traj, 12, out_dim]

        return logits_y, KLD

class DGCN(VAE):
    
    def __init__(self, net_params, **kwargs):
        super().__init__(net_params, **kwargs)
        self.name = 'DGCN'

        self.z_dim = net_params['z_dim']
        self.a_dim = net_params['a_dim']

        self.obsv_len = net_params['obsv_seq_len']
        self.pred_len = net_params['pred_seq_len']

        self.node_inputs = net_params['node_inputs']
        self.node_outputs = net_params['node_outputs']
        
        self.edge_inputs = net_params['edge_inputs']
        self.edge_outputs = net_params['edge_outputs']

        self.dt = net_params['dt']
        
        self.node_loss_wt = net_params['node_loss_wt']
        self.edge_loss_wt = net_params['edge_loss_wt']

        self.sample_timesteps = net_params['sample_timesteps'] 

        # input/output dimensions
        in_dim_node = {s:STATE_DIMS[s] for s in net_params['node_inputs']} 
        in_dim_edge = {s:STATE_DIMS[s] for s in net_params['edge_inputs']}

        out_dim_node =  {s:loss_func_param(net_params['node_loss'][s], STATE_DIMS[s]) for s in net_params['node_outputs']}
        out_dim_edge = {s:loss_func_param(net_params['edge_loss'][s], STATE_DIMS[s]) for s in net_params['edge_outputs']}
        
        ###### encoders #####
        #adjust input/output dimension for encoders
        for encoder in ['past_enc', 'target_enc']:
            net_params[encoder]['in_dim_node'] = in_dim_node
            net_params[encoder]['in_dim_edge'] = in_dim_edge
            net_params[encoder]['out_dim_node'] = out_dim_node
            net_params[encoder]['out_dim_edge'] = out_dim_edge

        net_params['past_enc'].update(net_params['net'])
        net_params['target_enc'].update(net_params['net'])
        
        self.past_enc = eval(net_params['net']['type'])(net_params['past_enc'])
        self.target_enc = eval(net_params['net']['type'])(net_params['target_enc'])

        ########### cvae ######
        self.cvae = nn.ModuleDict({'f_pz_x':MLP(net_params['enc_hdim'], net_params['z_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim'])),
                                  'f_pa_ex':MLP(sum(in_dim_edge.values()), net_params['a_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim'])),
                                  'f_qa_ex_ey':MLP(sum(in_dim_edge.values()), net_params['a_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim'])),
                                  'f_qz_x_y':MLP(net_params['enc_hdim'], net_params['z_dim']*2, hidden_size=(net_params['enc_hdim'], net_params['enc_hdim']))
                                  })

        ###### decoder ########
        self.traj_dec = nn.ModuleDict()
        
        latent_dims = (self.z_dim+self.a_dim)*(self.pred_len if self.sample_timesteps else 1)
        
        if net_params['traj_dec']['type']=='MLPDec':
            net_params['traj_dec']['in_dim'] = sum(in_dim_node.values()) + net_params['enc_hdim'] + latent_dims            
            for s in net_params['node_outputs']:
                net_params['traj_dec']['out_dim'] = net_params['pred_seq_len']*loss_func_param(net_params['node_loss'][s], STATE_DIMS[s])
                self.traj_dec[s] = MLPDec(net_params['traj_dec'], **kwargs)
                
        elif net_params['traj_dec']['type']=='RNNDec':
            net_params['traj_dec']['hidden_dim'] = net_params['dec_hdim']
            net_params['traj_dec']['in_dim'] = sum(out_dim_node.values()) + net_params['enc_hdim'] + latent_dims
            for s in net_params['node_outputs']:
                net_params['traj_dec']['out_dim'] = loss_func_param(net_params['node_loss'][s], STATE_DIMS[s])
                self.traj_dec[s] = RNNDec(net_params['traj_dec'], **kwargs)
        else:
            raise Exception(f"{net_params['net']['type']} not a valid dec type" )


    def forward(self, gx, xx, ex, gy=None, yy=None, ey=None, gt=None, comm_traj=None, num_samples=1, device=torch.device('cpu')):
        
        xx_in = torch.cat([v for _, v in xx.items()], dim=-1)
        ex_in = torch.cat([v for _, v in ex.items()], dim=-1)
        if ey is not None:
            ey_in = torch.cat([v for _, v in ey.items()], dim=-1)
        
        x0 = torch.stack([xx_in[gx.ndata['tid']==tid][-1] for tid in comm_traj], dim=0) #[num_traj, 6]

        #Encode X~p(X)
        gx, _, _ = self.past_enc(gx, xx, ex)
        
        h_x = torch.stack([torch.mean(gx.ndata['h'][gx.ndata['tid']==tid], dim=0) for tid in comm_traj], dim=0) # [num_traj, hidden_dim]                
        
        pz_x =  self.p_dist_from_h(self.cvae['f_pz_x'](h_x), self.z_dim, p_dist=self.pz_x) # p_distribution

        h_ex, _ = get_neighbors_edata(gx, traj_ids=comm_traj, edge_data=ex_in, aggregation='average')
        
        pa_ex = self.p_dist_from_h(self.cvae['f_pa_ex'](h_ex), self.a_dim, p_dist=self.pa_ex)

        KLD = 0
        sample_size = (num_samples, self.pred_len) if self.sample_timesteps else (num_samples, ) #(num_samples, 12, num_traj, a_dim) or (num_samples, num_traj, a_dim)    

        if self.training:
            #Encode X, Y ~ q(X, Y)
            gy, _, _ = self.target_enc(gy, yy, ey)

            h_x_y = torch.stack([torch.mean(gy.ndata['h'][gy.ndata['tid']==tid], dim=0) for tid in comm_traj], dim=0)
            # h_x_y = torch.cat([h_x, h_y], dim=-1)
            qz_x_y = self.q_dist_from_h(self.cvae['f_qz_x_y'](h_x_y), self.z_dim, self.qz_x_y)

            h_ex_ey, _ = get_neighbors_edata(gy, traj_ids=comm_traj, edge_data=ey_in, aggregation='average')
            
            # h_ex_ey = torch.cat([h_ex, h_ey], dim=-1)
            qa_ex_ey = self.q_dist_from_h(self.cvae['f_qa_ex_ey'](h_ex_ey), self.a_dim, self.qa_ex_ey)
            
            z = qz_x_y.rsample(sample_size) # (num_samples, 12, num_traj, z_dim) or (num_samples, num_traj, z_dim)          
            a = qa_ex_ey.rsample(sample_size) # (num_samples, 12, num_traj, a_dim) or (num_samples, num_traj, a_dim)   

            KLD = self.kl_divergence(qz_x_y, pz_x, samples=z) + self.kl_divergence(qa_ex_ey, pa_ex, samples=a)
            
        else:
            
            a = pa_ex.sample(sample_size) 
            z = pz_x.sample(sample_size) 
        
        self.a = a
        self.z = z

        # reshape latent they are of shape  (num_samples, 12, num_traj, z_dim)
        if self.sample_timesteps:
            z = z.permute(0, 2, 1, 3).reshape(num_samples, len(comm_traj), self.pred_len*self.z_dim) 
            a = a.permute(0, 2, 1, 3).reshape(num_samples, len(comm_traj), self.pred_len*self.a_dim)

        za = torch.cat([z.view(num_samples*len(comm_traj), -1),
                        a.view(num_samples*len(comm_traj), -1),
                        ], dim=-1) #[num_samples*num_traj, h_dim + z_dim + input_dim]
        
        h_x = h_x.repeat(num_samples, 1).view(num_samples*len(comm_traj), -1)
        
        x0 = x0.repeat(num_samples, 1).view(num_samples*len(comm_traj), -1)
        
        logits_y = {s:self.traj_dec[s](za, h_x, x0, num_samples=num_samples, device=device) for s in self.node_outputs}

        logits_y = {s:logits_y[s].view(num_samples, len(comm_traj), self.pred_len, -1) for s in self.node_outputs}#[num_samples, num_traj, 12, 2]
        
        return logits_y, KLD


def gnn_model(model_name, model_params):
    
    gnn_models = {'GCN':GCN,
              'DGCN': DGCN,
              'STDec': STDec,
              'GoalDec':GoalDec
              }
    return gnn_models[model_name](model_params)
