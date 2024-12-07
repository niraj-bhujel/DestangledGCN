_base_ = ['./base/pedestrians.py']

model = 'STDec'

K = 1
K_eval = 20
augment_prob = 0.0

hidden_dim = 128
num_layers = 1
min_pred_len = 6

node_attrs = ['ntx',]
node_inputs = node_attrs + ['vel', 'vnorm', 'dir']
node_outputs = ['vel', 'hed',] # node outputs will be used to check condition for computing node losses/readout instead of edge_loss_wt
node_loss = {'vel':'smoothl1_loss', 'hed':'smoothl1_loss', 'pos':'mv_normal'}
node_loss_wt = {'vel':0.05, 'hed':0.001, 'pos':0.0001}

edge_inputs = ['diff', 'dist', 'dir']
edge_outputs = ['dist'] # edge outputs will be used to check condition for computing edge losses and edge readout instead of edge_loss_wt
edge_loss = {'dist':'smoothl1_loss',}
edge_loss_wt = {'dist':0.05,}

learn_loss_weights = False

sample_goal = 0

net = dict(type = 'GCNNetSF',
        layer = 'GatedGCNLayerSF',
        dropout = 0.,
        batch_norm = False,
        residual = True,
        activation = 'ReLU',
        pos_enc = False,
        mlp_readout_node = True,
        mlp_readout_edge = False,
        )

trajdec = 'ConstVel'

