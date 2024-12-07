#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gatedgcn_layer import GatedGCNLayer
from layers.mlp_layer import MLPReadout, MLP


class SAGCNNet(nn.Module):
    
    def __init__(self, net_params):
        
        super().__init__()

        in_dim_node = net_params['in_dim_node']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        num_layers = net_params['num_layers']
        num_heads = net_params['num_heads']
        out_dim_node = net_params['out_dim_node']
        out_dim_edge = net_params['out_dim_edge']
        attn_dropout = net_params['attn_dropout']
        dropout = net_params['dropout']
        activation = getattr(nn, net_params['activation'])()
        
        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']

        self.batch_norm = net_params['batch_norm']

        self.residual = net_params['residual']
        self.layer = net_params['layer']

        self.pos_enc = net_params['pos_enc']
        
        self.batch_first = False # backward compatibility

        if self.layer=='GatedGCNLayer':
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
            
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
            

        if self.pos_enc:
            self.embedding_p = nn.Linear(net_params['pos_enc_dim'], hidden_dim)
            self.I_hp = nn.Linear(hidden_dim*2, hidden_dim)

        self.local_layers = nn.ModuleList([eval(self.layer)(input_dim=hidden_dim,
                                           output_dim=hidden_dim,
                                           dropout=dropout,
                                           batch_norm=self.batch_norm,
                                           residual=self.residual) for _ in range(num_layers)])

        self.global_layers = nn.ModuleList([nn.MultiheadAttention(hidden_dim,
                                                                    num_heads=num_heads, 
                                                                    dropout=attn_dropout,
                                                                    ) for _ in range(num_layers)])

        if self.batch_norm:
            self.bn_att = nn.BatchNorm1d(hidden_dim)


        self.ffn_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim*2, hidden_dim),
                                        nn.Dropout(dropout),
                                        )
        if self.batch_norm:
            self.bn_ffn = nn.BatchNorm1d(hidden_dim)

        if self.mlp_readout_node:
            self.MLP_nodes = MLPReadout(hidden_dim, out_dim_node, activation)
            # self.MLP_nodes = MLP(hidden_dim, out_dim_node, hidden_size=(hidden_dim, hidden_dim), activation=self.activation)
            
        if self.mlp_readout_edge:
            self.MLP_edges = MLPReadout(hidden_dim if self.layer=='GatedGCNLayer' else hidden_dim*2, self.out_dim_edge, self.activation)
            # self.MLP_edges = nn.Linear(hidden_dim*2, self.out_dim_edge, self.activation)

    def _sa_block(self, sa_layer, x, attn_mask, key_padding_mask):
        
        x = sa_layer(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       need_weights=False)[0]

        if self.batch_first:
            x = x[~key_padding_mask]
        else:
            x = x.permute(1, 0, 2)[~key_padding_mask]

        return x

    def forward(self, g, h, e, p=None):
    
        # input embedding
        h = self.embedding_h(h)
        if self.layer=='GatedGCNLayer':
            e = self.embedding_e(e)
        
        if self.pos_enc:
            p = g.ndata['pos_enc'].double()
            p = self.embedding_p(p)
            
        for i in range(len(self.local_layers)):
        # for gcn in self.local_layers:
            if self.pos_enc:
                h = torch.cat([h, p], dim=-1)
                h = self.I_hp(h)

            h_local, e, _ = self.local_layers[i](g, h, e, p)
            
            h_dense, mask = _to_dense_tensor(h, g.batch_num_nodes())

            h_global = self._sa_block(self.global_layers[i], h_dense, None, ~mask)

            # h_global = h + h_global # residual connection

            if self.batch_norm:
                h_global = self.bn_att(h_global)

            h = sum([h_local, h_global]) # sum 
            
            h = h + self.ffn_layer(h)

            if self.batch_norm:
                h = self.bn_ffn(h)

        # update graph to use h and e after bn and activations  during MLPEdge readout
        g.ndata['h'] = h
        g.edata['e'] = e

        # node output
        if self.mlp_readout_node:
            h = self.MLP_nodes(h)

        #edge output
        if self.mlp_readout_edge:
            e = self.MLP_edges(e)
            
        return g, h, e

def _to_dense_tensor(x, batch_num_nodes, fill_value=0., batch_first=False):
        
    batch_size = len(batch_num_nodes)
    
    cum_nodes = torch.cat([batch_num_nodes.new_zeros(1), batch_num_nodes.cumsum(dim=0)])
    
    max_num_nodes = int(batch_num_nodes.max())

    batch = torch.cat([torch.tensor(b).repeat(num_nodes) for b, num_nodes in enumerate(batch_num_nodes)]).to(batch_num_nodes.device)

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    feature_dims = list(x.size())[1:]
    size = [batch_size * max_num_nodes] + feature_dims 
    out = x.new_full(size, fill_value)
    out[idx] = x

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=x.device)
    mask[idx] = 1

    mask = mask.view(batch_size, max_num_nodes) # MultiHeadAttention expects key_padding_mask.shape == (bs, src_len)

    if batch_first:
        out = out.view([batch_size, max_num_nodes] + feature_dims)
    else:
        out = out.view([max_num_nodes, batch_size] + feature_dims)
    
    return out, mask