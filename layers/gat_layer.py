import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GATLayer(nn.Module):
    """
    Parameters
    ----------
    input_dim : 
        Number of input features.
    output_dim : 
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        
    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """    
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, activation, **kwargs):
        super().__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        self.activation = getattr(torch.nn, activation)()
        num_heads = kwargs['num_heads']

        self.gatconv = GATConv(input_dim, output_dim, num_heads, dropout, kwargs['attn_dropout'], allow_zero_in_degree=True)

        self.out_proj = nn.Linear(output_dim*num_heads, output_dim)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(output_dim * num_heads)

    def forward(self, g, h, e=None, p=None):
        h_in = h # for residual connection

        h = self.gatconv(g, h)

        h = self.out_proj(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)
            
        if self.activation:
            h = self.activation(h)
            
        if self.residual:
            h = h_in + h # residual connection

        return h, e, p
    

##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class CustomGATHeadLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_fc = nn.Linear(2 * output_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(output_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayer(nn.Module):
    """
        Param: [input_dim, output_dim, n_heads]
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, **kwargs):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.residual = residual
        self.num_heads = kwargs['num_heads']
        self.attn_dropout = kwargs['attn_dropout']

        if input_dim != (output_dim*self.num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.heads.append(CustomGATHeadLayer(input_dim, output_dim, self.attn_dropout, batch_norm))
        
        self.merge = 'max' 

    def forward(self, g, h, e, p=None):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'max':
            h = torch.stack(head_outs).max(0).values
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e, p
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={}, attn_dropout={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, 
                                             self.num_heads,
                                              self.residual,
                                              self.attn_dropout)

    
##############################################################


class GATHeadLayerEdge(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc_h = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_proj = nn.Linear(3* output_dim, output_dim)
        self.attn_fc = nn.Linear(3* output_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(output_dim)
        self.batchnorm_e = nn.BatchNorm1d(output_dim)

    def edge_attention(self, edges):
        z = torch.cat([edges.data['z_e'], edges.src['z_h'], edges.dst['z_h']], dim=1)
        e_proj = self.fc_proj(z)
        attn = F.leaky_relu(self.attn_fc(z))
        return {'attn': attn, 'e_proj': e_proj}

    def message_func(self, edges):
        return {'z': edges.src['z_h'], 'attn': edges.data['attn']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}
    
    def forward(self, g, h, e):
        z_h = self.fc_h(h)
        z_e = self.fc_e(e)
        g.ndata['z_h'] = z_h
        g.edata['z_e'] = z_e
        
        g.apply_edges(self.edge_attention)
        
        g.update_all(self.message_func, self.reduce_func)
        
        h = g.ndata['h']
        e = g.edata['e_proj']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        h = F.elu(h)
        e = F.elu(e)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    

class CustomGATLayerEdge(nn.Module):
    """
        Param: [input_dim, output_dim, n_heads]
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, activation, **kwargs):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.residual = residual
        self.num_heads = kwargs['num_heads']
        self.attn_dropout = kwargs['attn_dropout']
        
        if input_dim != (output_dim*self.num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.heads.append(GATHeadLayerEdge(input_dim, output_dim, self.attn_dropout, batch_norm))
        self.merge = 'sum' 

    def forward(self, g, h, e, p=None):
        h_in = h # for residual connection
        e_in = e

        head_outs_h = []
        head_outs_e = []
        for attn_head in self.heads:
            h_temp, e_temp = attn_head(g, h, e)
            head_outs_h.append(h_temp)
            head_outs_e.append(e_temp)

        if self.merge == 'cat':
            h = torch.cat(head_outs_h, dim=1)
            e = torch.cat(head_outs_e, dim=1)
        elif self.merge=='sum':
            h = torch.stack(head_outs_h, dim=0).sum(0)
        else:
            raise NotImplementedError

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e

        return h, e, p
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={}, att_dropout={})'.format(self.__class__.__name__,
                                             self.in_channels,self.out_channels, self.num_heads, self.residual, self.attn_dropout)

    
##############################################################


class GATHeadLayerIsotropic(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        return {'z': edges.src['z']}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class CustomGATLayerIsotropic(nn.Module):
    """
        Param: [input_dim, output_dim, n_heads]
    """
    def __init__(self, input_dim, output_dim, num_heads, dropout, batch_norm, residual=True):
        super().__init__()

        self.in_channels = input_dim
        self.out_channels = output_dim
        self.num_heads = num_heads
        self.residual = residual

        if input_dim != (output_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATHeadLayerIsotropic(input_dim, output_dim, dropout, batch_norm))
        self.merge = 'cat' 

    def forward(self, g, h, e):
        h_in = h # for residual connection
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
