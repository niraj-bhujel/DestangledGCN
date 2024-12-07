#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from layers.gatedgcn_layer import *
from layers.gat_layer import *
from layers.gatedgcn_lspe_layer import GatedGCNLSPELayer
from layers.mlp_layer import MLPReadout, MLP

class GCNNet(nn.Module):
    
    def __init__(self, net_params, **kwargs):
        
        super().__init__()

        # self.dt = net_params['dt']
        self.node_inputs = list(net_params['in_dim_node'])
        self.edge_inputs = list(net_params['in_dim_edge'])
        
        in_dim_node = sum(net_params['in_dim_node'].values())
        in_dim_edge = sum(net_params['in_dim_edge'].values())

        layer = net_params['layer']
        hidden_dim = net_params['hidden_dim']

        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']

        if self.mlp_readout_node:
            self.node_outputs = list(net_params['out_dim_node'])
            
        if self.mlp_readout_edge:
            self.edge_outputs = list(net_params['out_dim_edge'])
            
        self.pos_enc = layer == 'GatedGCNLSPELayer'
        self.embed_edge = layer in ['GatedGCNLayer', 'CustomGatedGCNLayer', 'GatedGCNLSPELayer'] 

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        # self.embedding_p = nn.Linear(2, hidden_dim)

        if self.embed_edge:
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        if self.pos_enc:
            self.embedding_p = nn.Linear(2, hidden_dim)

        kwargs = dict(input_dim = hidden_dim, 
                    output_dim = hidden_dim, 
                    dropout = net_params['dropout'], 
                    batch_norm = net_params['batch_norm'], 
                    residual = net_params['residual'],
                    activation = net_params['activation'],
                    )

        if 'GAT' in str(layer):
            kwargs.update(dict(num_heads=net_params['num_heads'],
                        attn_dropout=net_params['attn_dropout']))

        self.layers = nn.ModuleList([eval(layer)(**kwargs) for _ in range(net_params['num_layers'])])
        
        if self.mlp_readout_node:
            self.MLP_nodes = nn.ModuleDict({s:MLPReadout(hidden_dim, net_params['out_dim_node'][s], net_params['activation'])
                                            for s in self.node_outputs})
            
        if self.mlp_readout_edge:
            self.MLP_edges = nn.ModuleDict({s:MLPReadout(hidden_dim if self.embed_edge else hidden_dim*2, net_params['out_dim_edge'][s], net_params['activation'])
                                            for s in self.edge_outputs})
            
    def forward(self, g, h, e, p=None):
        h_in = h
        e_in = e

        h = torch.cat([h[k] for k in self.node_inputs], dim=-1)
        h = self.embedding_h(h)

        e = torch.cat([e[k] for k in self.edge_inputs], dim=-1)
        if self.embed_edge:
            e = self.embedding_e(e)

        if self.pos_enc:
            # p = g.ndata['pos_enc'].double()
            p = self.embedding_p(h_in['pos'])

        for i, conv in enumerate(self.layers):
            # print('CONV Layer:', i)
            h, e, p = conv(g, h, e, p)

        # # # update graph to use h and e after bn and activations  during MLPEdge readout
        g.ndata['h'] = h
        g.edata['e'] = e

        e_out = {'e': e}
        h_out = {'vel': h, 'pos':p}

        # node output
        if self.mlp_readout_node:
            h_out = {s:self.MLP_nodes[s](h_out[s]) for s in self.node_outputs}

        #edge output
        if self.mlp_readout_edge:
            if not self.embed_edge:
                def _edge_feat(edges): return {'e':torch.cat([edges.src['h'], edges.dst['h']], dim=-1)}
                g.apply_edges(_edge_feat)
                e_out = g.edata['e']
                
            e_out = {s:self.MLP_edges[s](e_out['e']) for s in self.edge_outputs}

        return g, h_out, e_out


class GCNNetSF(nn.Module):
    
    def __init__(self, net_params, **kwargs):
        
        super().__init__()
        # self.dt = kwargs['dt']
        self.node_inputs = list(net_params['in_dim_node'])
        self.edge_inputs = list(net_params['in_dim_edge'])
        
        self.node_outputs = list(net_params['out_dim_node'])
        self.edge_outputs = list(net_params['out_dim_edge'])

        in_dim_node = sum(net_params['in_dim_node'].values())
        in_dim_edge = sum(net_params['in_dim_edge'].values())

        layer = net_params['layer']
        hidden_dim = net_params['hidden_dim']

        self.layer_name = layer
        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']
        
        self.embed_edge = layer in ['GatedGCNLayer', 'CustomGatedGCNLayer', 'GatedGCNLayerSF', 'GatedGCNLSPELayer'] 

        self.embedding_h =  nn.Linear(in_dim_node, hidden_dim)
        self.embedding_p = nn.Linear(2, hidden_dim)
        self.embedding_d = nn.Linear(2, hidden_dim)

        if self.embed_edge:
            self.embedding_e =  nn.Linear(in_dim_edge, hidden_dim) 

        kwargs = dict(input_dim = hidden_dim, 
                    output_dim = hidden_dim, 
                    dropout = net_params['dropout'], 
                    batch_norm = net_params['batch_norm'], 
                    residual = net_params['residual'],
                    activation = net_params['activation'],
                    )

        if 'GAT' in str(layer):
            kwargs.update(dict(num_heads=net_params['num_heads'],
                        attn_dropout=net_params['attn_dropout']))

        self.layers = nn.ModuleList([eval(layer)(**kwargs) for _ in range(net_params['num_layers'])])
        
        if self.mlp_readout_node:
            self.MLP_nodes = nn.ModuleDict({s:MLPReadout(hidden_dim, net_params['out_dim_node'][s], net_params['activation'])
                                            for s in self.node_outputs})
            
        if self.mlp_readout_edge:
            self.MLP_edges = nn.ModuleDict({s:MLPReadout(hidden_dim if self.embed_edge else hidden_dim*2, net_params['out_dim_edge'][s], net_params['activation'])
                                            for s in self.edge_outputs})

        # self.V = nn.Linear(hidden_dim, hidden_dim)
    
    def value_rab(self, rab, sigma=0.3):
        # rab = torch.linalg.norm(rab, dim=-1, keepdim=True)
        rab = 0.5 * torch.sqrt(torch.linalg.norm(rab, ord=2, dim=-1, keepdim=True).clamp(min=1e-9))  
        return torch.exp(-rab/sigma)

    def message_function_vab(self, res='Vj'):
        def msg_func(edges):
            # hj = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
            hj = edges.src['h'] * edges.dst['h']
            return {res: F.relu(self.V(hj))}
        return msg_func

    def forward(self, g, h, e):

        p = self.embedding_p(h['pos'])
        d = self.embedding_d(h['hed'].round() * h['speed'])

        h = torch.cat([h[k] for k in self.node_inputs], dim=-1)
        h = self.embedding_h(h)

        e = torch.cat([e[k] for k in self.edge_inputs], dim=-1)
        if self.embed_edge:
            e = self.embedding_e(e)

        h_in, e_in, p_in, d_in = h, e, p, d
        for i, conv in enumerate(self.layers):
            # print('CONV Layer:', i)
            h, e, p, d = conv(g, h, e, p=p, d=d)

        g.ndata['h'] = h
        g.edata['e'] = e

        # spatial_edges = g.edges(form='eid')[g.edata['spatial_mask'].flatten()==1]
        # temporal_edges = g.edges(form='eid')[g.edata['spatial_mask'].flatten()==0]
        # # compute f0
        # f0 = 1/0.5*(d - h)
        # f = f0
        # # compute f_ab
        # if len(spatial_edges)>0:
        #     # print('Computing fab')
        #     def compute(rab):              
        #         g.edata['b']= self.value_rab(rab)
        #         g.apply_edges(self.message_function_vab('Vj'), edges=spatial_edges)
        #         g.apply_edges(e_mul_e('Vj', 'b', 'V_ab'), edges=spatial_edges)
        #         return g.edata['V_ab']
        #     with torch.enable_grad():
        #         vector = torch.ones_like(e, requires_grad=False)
        #         g.edata['grad_rab'] = torch.autograd.functional.vjp(compute, e, vector, create_graph=True, strict=True)[1]
        #     g.update_all(fn.copy_e('grad_rab', 'm'), fn.sum('m', 'fab'))
        #     f = f0 - g.ndata['fab']

        # # update nodes
        # h = h + f*self.dt
        # # d = d_in + f/torch.linalg.norm(f, dim=-1, keepdim=True).add(1e-9)

        # else:
        h_out = {'vel': h, 'pos':p, 'hed':d}
        e_out = {'e': e}
        # node output
        if self.mlp_readout_node:
            h_out = {s:self.MLP_nodes[s](h_out[s]) for s in self.node_outputs}

        #edge output
        if self.mlp_readout_edge:
            if not self.embed_edge:
                def _edge_feat(edges): return {'e':torch.cat([edges.src['h'], edges.dst['h']], dim=-1)}
                g.apply_edges(_edge_feat)
                e_out = g.edata['e']
                
            e_out = {s:self.MLP_edges[s](e_out['e']) for s in self.edge_outputs}

        return g, h_out, e_out