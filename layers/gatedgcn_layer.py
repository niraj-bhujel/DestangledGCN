import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn.utils import spectral_norm
"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

def e_mul_e(src, dst, res):
    def msg_func(edges):return {res: edges.data[src] * edges.data[dst]}
    return msg_func

def e_add_e(src, dst, res):
    def msg_func(edges):return {res: edges.data[src] + edges.data[dst]}
    return msg_func

def e_div_e(src, dst, res):
    def msg_func(edges):return {res: edges.data[src]/(edges.data[dst]+1e-9)}
    return msg_func

class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, h, e, p=None,):
        
        if self.residual:
            h_in = h # for residual connection
            e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
            
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, p
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, batch_norm={}, residual={}, dropout={})'.format(
            self.__class__.__name__,
             self.in_channels,
             self.out_channels, 
             self.batch_norm,
             self.residual,
             self.dropout)

class CustomGatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, bias=True, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.bias = bias
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=bias)
        self.B = nn.Linear(input_dim, output_dim, bias=bias)
        self.C = nn.Linear(input_dim, output_dim, bias=bias)
        self.D = nn.Linear(input_dim, output_dim, bias=bias)
        self.E = nn.Linear(input_dim, output_dim, bias=bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)

    def e_mul_e(self, src, dst, res):
        def _func(edges):return {res: edges.data[src] * edges.data[dst]}
        return _func

    def e_add_e(self, src, dst, res):
        def _func(edges):return {res: edges.data[src] + edges.data[dst]}
        return _func

    def forward(self, g, h, e, p=None,):
        
        if self.residual:
            h_in = h # for residual connection
            e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e)

        edge_ids = g.edges(form='eid')
        spatial_eids = edge_ids[g.edata['spatial_mask'].flatten()==1]
        temporal_eids = edge_ids[g.edata['spatial_mask'].flatten()==0]

        # edge feature updates
        g.apply_edges(fn.v_sub_u('Dh', 'Eh', 'DEh')) # d_ij = h_j - h_i
        g.apply_edges(fn.u_add_e('Bh', 'DEh', 'he')) # h_j = h_i + d_{ij}

        # update edge
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce'] # d_ij^(l) = d_ij^(l-1) + C.d_ij^(l-1)
        # g.edata['sigma'] = torch.sigmoid(g.edata['e'])

        # zalungo circular specification
        g.edata['e_norm'] = 0.5*torch.linalg.norm(g.edata['e'], dim=-1, keepdim=True)
        g.edata['sigma'] = g.edata['e_norm'].mul(-1).exp() # replace sigmoid (saturation problem) with sigma= exp(-d_ij/2_\sigma^2) with \sigma^2 set to 1
        # g.edata['e_dir'] = torch.where(g.edata['e_norm']>0, g.edata['e']/g.edata['e_norm'], g.edata['e'])
        # g.edata['r'] = g.edata['sigma'] * torch.where(g.edata['e_norm']>0, g.edata['e']/g.edata['e_norm'], g.edata['e'])

        # yamaguchi attraction, 
        # g.ndata['h_dir'] = g.ndata['dir']
        g.edata['e_dir'] = torch.where(g.edata['dist']>0, g.edata['diff']/g.edata['dist'], g.edata['diff'])
        g.apply_edges(fn.u_dot_v('dir', 'dir', 'uv_dir')) # first term (flip direction if opposite direction)
        g.apply_edges(fn.e_dot_v('e_dir', 'dir', 'eh_dir')) # second term (less weight if j is outside the view of ped i)
        g.edata['a'] = 0.5*(1-g.edata['eh_dir']) * g.edata['uv_dir'] 
        # g.edata['a'] = torch.tanh(g.edata['eh_dir'] * g.edata['uv_dir'])
        # g.edata['a'] = torch.sigmoid(g.edata['eh_dir']) * g.edata['uv_dir']

        g.edata['f'] = g.edata['a'] * g.edata['sigma']

        # update for spatial and temporal edges
        if len(spatial_eids)>0:
            # g.apply_edges(fn.u_mul_e('Bh', 'f', 'h_sigma'), edges=spatial_eids)
            g.apply_edges(self.e_mul_e('he', 'f', 'h_sigma'), edges=spatial_eids)

        if len(temporal_eids)>0:
            # g.apply_edges(fn.u_mul_e('Bh', 'sigma', 'h_sigma'), edges=temporal_eids)
            g.apply_edges(self.e_mul_e('he', 'sigma', 'h_sigma'), edges=temporal_eids)

        g.update_all(fn.copy_e('h_sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h']
        
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
            
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation


        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, p
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, bias={}, batch_norm={}, residual={}, dropout={})'.format(
            self.__class__.__name__,
             self.in_channels,
             self.out_channels,
             self.bias, 
             self.batch_norm,
             self.residual,
             self.dropout)

class GatedGCNLayerSF(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual, bias=True, **kwargs):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.bias = bias

        if input_dim != output_dim:
            self.residual = False

        self.V1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.V2 = nn.Linear(input_dim, output_dim, bias=bias)

        self.E1 = nn.Linear(input_dim, output_dim, bias=bias)

        # self.A = nn.Linear(input_dim, output_dim, bias=bias)
        self.P1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.P2 = nn.Linear(input_dim, output_dim, bias=bias)
        self.P3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.D1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.D2 = nn.Linear(input_dim, output_dim, bias=bias)

        self.T = nn.Linear(input_dim*2, output_dim, bias=bias)
        self.V = nn.Linear(input_dim, output_dim, bias=bias)
        # self.R = nn.Linear(input_dim, output_dim, bias=bias)


        self.V1 = spectral_norm(self.V1)
        self.V2 = spectral_norm(self.V2)
        self.E1 = spectral_norm(self.E1)
        self.P1 = spectral_norm(self.P1)
        self.P2 = spectral_norm(self.P2)
        self.P3 = spectral_norm(self.P3)
        self.D1 = spectral_norm(self.D1)
        self.D2 = spectral_norm(self.D2)
        self.V = spectral_norm(self.V)
        self.T = spectral_norm(self.T)
        # self.F = spectral_norm(self.F)
        # self.R = spectral_norm(self.R)

    
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

    def message_function_tau(self, res='tau'):
        def msg_func(edges):
            hi = torch.cat([edges.dst['h'], edges.dst['d']], dim=-1)
            return {res: F.relu(self.T(hi))}
        return msg_func

    def message_function_hf(self, src='h', res='hf'):
        def msg_func(edges):
            hj = (edges.dst['p'] - edges.src['p'])/0.4
            return {res: hj}
        return msg_func

    def message_function_pf(self, src='p', res='pf'):
        def msg_func(edges):
            hj = edges.src[src] + 0.4*edges.src['f'] + 0.5*edges.src['f']*0.4**2
            return {res: hj}
        return msg_func

    def normalize_grad_rab(self, src='e', dst='e', res='norm_f'):
        def msg_func(edges):
            return {res: edges.data[src] / (edges.dst[dst] + 1e-6)} # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
        return msg_func

    def forward(self, g, h, e, p=None, d=None):

        if self.residual:
            h_in = h # for residual connection
            e_in = e # for residual connection
            p_in = p
            d_in = d
        
        g.ndata['h'] = h
        g.ndata['V1h'] = self.V1(h)
        g.ndata['V2h'] = self.V2(h)

        g.edata['e']  = e 
        g.edata['E1e'] = self.E1(e) 

        g.ndata['p'] = p
        g.ndata['P1h'] = self.P1(p) # init pos
        g.ndata['P2h'] = self.P2(p) # src pos
        g.ndata['P3h'] = self.P3(p) # dst pos

        g.ndata['d'] = d
        g.ndata['D1h'] = self.D1(d) # desired vel
        g.ndata['D2h'] = self.D2(d) # desired vel


        # update edges
        # g.apply_edges(self.message_function_edge('p', 'p', 'PPh'))
        # g.edata['e'] = g.edata['PPh'] + g.edata['Ee']
        g.apply_edges(fn.v_sub_u('P1h', 'P2h', 'PPh')) # d_ij(l) = h_j(l) - h_i(l)
        g.edata['e'] = 0.5*(g.edata['PPh'] + g.edata['E1e']) # d_ij^(l) = d_ij^(l-1) + W(d_ij^(l-1))
        # g.edata['sigma'] = torch.sigmoid(g.edata['e'])

        spatial_edges = g.edges(form='eid')[g.edata['spatial_mask'].flatten()==1]
        temporal_edges = g.edges(form='eid')[g.edata['spatial_mask'].flatten()==0]

        # compute f0
        # g.ndata['a'] = g.ndata['d'] - g.ndata['h']
        g.ndata['a'] = g.ndata['D1h'] - g.ndata['V1h']
        g.apply_edges(self.message_function_tau('tau'), edges=temporal_edges)
        g.update_all(fn.copy_e('tau', 'm'), fn.sum('m', 'sum_tau'))
        g.ndata['f0'] = g.ndata['a'] * g.ndata['sum_tau']
        # g.ndata['f0'] = g.ndata['a'] / 0.5
        g.ndata['f'] = g.ndata['f0']
        
        # compute f_ab
        if len(spatial_edges)>0:
            # print('Computing fab')
            def compute(rab):              
                g.edata['b']= self.value_rab(rab)
                g.apply_edges(self.message_function_vab('Vj'), edges=spatial_edges)
                g.apply_edges(e_mul_e('Vj', 'b', 'V_ab'), edges=spatial_edges)
                return g.edata['V_ab']
            with torch.enable_grad():
                rab = g.edata['e']
                # rab = torch.sigmoid(rab)
                # rab = self.R(rab)
                vector = torch.ones_like(rab, requires_grad=False)
                g.edata['grad_rab'] = torch.autograd.functional.vjp(compute, rab, vector, create_graph=True, strict=True)[1]
                # print(g.edata['grad_rab'].shape)
            g.update_all(fn.copy_e('grad_rab', 'm'), fn.sum('m', 'fab'))
            g.ndata['f'] = g.ndata['f0'] - g.ndata['fab']

        # update nodes
        g.ndata['h'] = g.ndata['V2h'] + g.ndata['f']*g.ndata['dt'].unsqueeze(1)
        g.ndata['p'] = g.ndata['P3h'] + g.ndata['f']*g.ndata['dt'].unsqueeze(1) + 0.5*g.ndata['f']*g.ndata['dt'].unsqueeze(1)**2
        g.ndata['d'] = g.ndata['D2h'] + g.ndata['f']/torch.linalg.norm(g.ndata['f'], dim=-1, keepdim=True).add(1e-9)

        # g.apply_edges(self.message_function_pf('P3h', 'pf'), edges=temporal_edges)
        # g.update_all(fn.copy_e('pf', 'm'), fn.sum('m', 'p'))

        # g.apply_edges(self.message_function_hf('p', 'hf'), edges=temporal_edges)
        # g.update_all(fn.u_add_e('V2h', 'hf', 'm'), fn.sum('m', 'h'))

        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        p = g.ndata['p']
        d = g.ndata['d']
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
            
        h = F.leaky_relu(h) # non-linear activation
        e = F.leaky_relu(e) # non-linear activation
        p = F.leaky_relu(p)
        d = F.leaky_relu(d)

        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
            p = p_in + p
            d = d_in + d

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)


        return h, e, p, d
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, bias={}, batch_norm={}, residual={}, dropout={})'.format(
            self.__class__.__name__,
             self.in_channels,
             self.out_channels,
             self.bias, 
             self.batch_norm,
             self.residual,
             self.dropout)