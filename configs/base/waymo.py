# dataset
dataset='waymo'
data_dir = os.path.join(os.getenv('ROOT_DIR', '..'), '/datasets/waymo/processed_data/')

version='full_10'
node_types=['PEDESTRIAN']

obsv_seq_len = 3
pred_seq_len = 16

min_obsv_len = 2
min_seq_len = 2
min_agents = 1

skip = 0
dt = 0.5
augment_prob = 0.5

include_lane = False

seed = 4309

# optimizer
lr = 0.0003
weight_decay = 0.05
grad_clip = 10
auto_scale_lr = False

# scheduler
lr_patience = 50
lr_reduce_factor = 0.9
lr_scheduler_metric = 'val_loss'

# Early stopping
early_stop_patience = 20
early_stop_metric = 'test_ade'


# training
train_bs = 64
test_bs = 64
phases = ['train', 'val']
K = 25 # Number of samples used in training
K_eval = 10 # Number of samples used for evaluations'
epochs = 300
initial_epoch = 1 # starting counter for epochs
test_interval = 5
num_workers = 0

# input/output states
node_inputs = ['vel', 'vnorm', 'acc', 'anorm', 'dir']
node_outputs = ['vel']
edge_inputs = ['diff', 'dist']
edge_outputs = ['dist']

# loss
traj_loss = 'multivariate_normal'
node_loss = 'l2_loss'
edge_loss = 'collision_loss'
traj_loss_wt = 1.0
kld_loss_wt = 1.0 
node_loss_wt = 0.
edge_loss_wt = 0.

standardization = {
    'PEDESTRIAN': {
        'pos': {'x': {'mean': 800., 'std': 5500.}, 'y': {'mean': 1200., 'std': 7500.}},
        'rel': {'x': {'mean': 0., 'std': 5.}, 'y': {'mean': 0., 'std': 5.}},
        'vel': {'x': {'mean': 0., 'std': 2.}, 'y': {'mean': 0., 'std': 2.}},
        'acc': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'hed': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'dir': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'vnorm': {'x': {'mean': 0., 'std': 1.}},
        'anorm': {'x': {'mean': 0., 'std': 1.},},
        'yaw': {'x': {'mean': 0., 'std': 3.14}}

    },
    'VEHICLE': {
        'pos': {'x': {'mean': 0., 'std': 10.}, 'y': {'mean': 0., 'std': 10.}},
        'rel': {'x': {'mean': 0., 'std': 40.}, 'y': {'mean': 0., 'std': 40.}},
        'vel': {'x': {'mean': 0., 'std': 5.}, 'y': {'mean': 0., 'std': 5.}},
        'acc': {'x': {'mean': 0., 'std': 4.}, 'y': {'mean': 0., 'std': 4.}},
        'hed': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'dir': {'x': {'mean': 0., 'std': 1.}, 'y': {'mean': 0., 'std': 1.}},
        'vnorm': {'x': {'mean': 0., 'std': 5.}},
        'anorm': {'x': {'mean': 0., 'std': 4.}},
        'yaw': {'x': {'mean': 0., 'std': 3.14}}

    },
}

standardization['CYCLIST'] = standardization['VEHICLE'].copy()
standardization['LANE'] = standardization['VEHICLE'].copy()
__attr_standardization = {
        'ntx': {'x': {'mean': 8., 'std': 12.}},
        'tid': {'x': {'mean': 0., 'std': 10000.}},
        'nid': {'x': {'mean': 0., 'std': 1000.}},
    }

for _, __node_std_param in standardization.items():
    __node_std_param.update(__attr_standardization)

