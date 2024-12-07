_base_ = ['./base/pedestrians.py']

model = 'STDec'

K = 20
K_eval = 20
augment_prob = 0.0

hidden_dim = 128
num_layers = 4
min_pred_len = 6

node_attrs = ['ntx',]
node_inputs = node_attrs + ['vel', 'vnorm', 'dir', 'hed']
node_outputs = ['vel']
node_loss = {'vel':'smoothl1_loss', 'pos':'mv_normal'}
node_loss_wt = {'vel':1.0, 'pos':1e-6}

edge_inputs = ['diff', 'dist', 'dir']
edge_outputs = ['dist'] # edge outputs will be used to check condition for edge loss and edge readout instead of edge_loss_wt
edge_loss = {'dist':'smoothl1_loss',}
edge_loss_wt = {'dist':1e-3,}

sample_goal = 0
learn_loss_weights = False

net = dict(type = 'GCNNet',
        layer = 'GatedGCNLayer',
        dropout = 0.,
        batch_norm = False,
        residual = True,
        activation = 'ReLU',
        pos_enc = True, # only True for GatedGCNLSPELayer
        )

trajdec = 'run-3_DGCN_GatedGCNLayer_eh256_el3_K1_K_eval20_bs1_skip8_pred_vel1.0_bhujeln1'
# trajdec = 'run-3_DGCN_GatedGCNLayer_eh256_el3_K10_K_eval20_bs1_skip8_pred_vel1.0_bhujeln1'
# trajdec = None
