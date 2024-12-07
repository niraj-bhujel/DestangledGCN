import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 

"""
    SAN-GT and SAN-GT-LSPE
    
"""

from layers.san_layer import SANLayer
from layers.mlp_layer import MLPReadout


class SANNet(nn.Module):
    def __init__(self, net_params, use_bias=False):
        super().__init__()

        hidden_dim = net_params['hidden_dim']
        num_layers = net_params['num_layers']

        num_heads = net_params['num_heads']

        attn_dropout = net_params['attn_dropout']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        activation = net_params['activation']
        layer_norm = net_params['layer_norm']

        self.pos_enc = net_params['pos_enc']

        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']

        self.embedding_h = nn.Linear(net_params['in_dim_node'], hidden_dim)
        self.embedding_e = nn.Linear(net_params['in_dim_edge'], hidden_dim)

        if self.pos_enc:
            self.embedding_p = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList([SANLayer(hidden_dim, hidden_dim, num_heads, attn_dropout,
                                             layer_norm, batch_norm, residual, use_bias, self.pos_enc)
                                                for _ in range(num_layers)])

        if self.mlp_readout_node:
            self.MLP_nodes = MLPReadout(hidden_dim, net_params['out_dim_node'])   

        if self.mlp_readout_edge:
            self.MLP_edges = MLPReadout(hidden_dim, net_params['out_dim_edge'])   
        
    def forward(self, g, h, e, p=None):

        # if self.training:
        #     h[...,0:2] = h[...,0:2] + torch.normal(g.ndata['loc'], g.ndata['std'])*0.1
        
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        if self.pos_enc:
            p = g.ndata['pos_enc'].double()
            p = self.embedding_p(p)
        
        for conv in self.layers:
            h, e, p = conv(g, h, e, p)
        
        g.ndata['h'] = h
        g.edata['e'] = e

        if self.mlp_readout_node:
            h = self.MLP_nodes(h)

        #edge output
        if self.mlp_readout_edge:
            e = self.MLP_edges(e)
            
        return g, h, e

