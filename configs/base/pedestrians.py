# dataset
dataset='pedestrians'
dset_names = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
data_dir = '../datasets/pedestrians/'
version=''
node_types=['PEDESTRIAN']

obsv_seq_len = 8
pred_seq_len = 12

min_obsv_len = 2
min_seq_len = 2
min_pred_len = 6
min_agents = 1

skip = 8 # skips frames during training
dt = 0.4
augment_prob = 0.5

include_lane = False

seed = 4309

# optimizer
lr = 0.0003
weight_decay = 0.0
grad_clip = 10
auto_scale_lr = False

# scheduler
lr_patience = 5
lr_reduce_factor = 0.9
lr_scheduler_metric = 'val_loss'
lr_reduce_epoch = 30 # reduce learning rate after this epoch

# Early stopping
early_stop_patience = 40
early_stop_metric = 'test_ade'


# training
phases = ['train', 'test']

train_bs = 1
test_bs = 64
test_interval = 5

K = 10
K_eval = 20
sample_timesteps = False


node_attrs = ['ntx',]
node_inputs = node_attrs + ['vel', 'vnorm', 'acc', 'anorm', 'hed']
node_outputs = ['vel'] # node outputs will be used to check condition for computing node losses/readout instead of edge_loss_wt
node_loss = {'vel':'mv_normal', 'pos':'l2_loss'}
node_loss_wt = {'vel':1.0, 'pos':1e-3}

edge_inputs = ['diff', 'dist', 'dir', 'spatial_mask', 'temporal_mask']
edge_outputs = []
edge_loss_wt = {}

kld_loss_wt = 1.0

loss_mode = 'average'

learn_loss_weights = False


standardization = {
    'PEDESTRIAN': {
        'pos': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 5., 'std': 5.}},
        'rel': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'vel': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'acc': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'hed': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'dir': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'speed': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'rnorm': {'x': {'mean': 0., 'std': 1.}},
        'vnorm': {'x': {'mean': 0., 'std': 1.}},
        'anorm': {'x': {'mean': 0., 'std': 1.},},
        'yaw': {'x': {'mean': 0., 'std': 1}}

    },
}

__attr_standardization = {
        'ntx': {'x': {'mean': 8., 'std': 1.}},
        'tid': {'x': {'mean': 0., 'std': 10000.}},
        'nid': {'x': {'mean': 0., 'std': 1000.}},
        'fid': {'x': {'mean': 0., 'std': 10000.}},
    }

for _, __node_std_param in standardization.items():
    __node_std_param.update(__attr_standardization)

