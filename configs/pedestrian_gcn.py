_base_ = ['./base/pedestrians.py']

model = 'GCN'

z_dim = 32
a_dim = 32
enc_hdim = 256
dec_hdim = 256
enc_layers = 3
dec_layers = 3

K = 1
K_eval = 20
sample_timesteps = False

#encoder
past_enc = dict(
    in_dim_node = 2,
    in_dim_edge = 1,
    embed = True,
    mlp_readout_node = False,
    mlp_readout_edge = False,
    out_dim_node = 32, # this will not be used if mlp_readout_node = False
    out_dim_edge = 1,  # this will not be used if mlp_readout_edge = False
    hidden_dim = enc_hdim,
    num_layers = enc_layers,
    )

#target encoder
target_enc = dict(
    in_dim_node = 2,
    in_dim_edge = 1,
    embed = True,
    mlp_readout_node = False,
    mlp_readout_edge = False,
    out_dim_node = 32, # this will not be used if mlp_readout_node = False
    out_dim_edge = 1, # this will not be used if mlp_readout_edge = False
    hidden_dim = enc_hdim,
    num_layers = enc_layers,
    )

# encoder and target network
net = dict(type = 'GCNNet',
        layer = 'GatedGCNLayer',
        dropout = 0.,
        batch_norm = False,
        residual = True,
        activation = 'ReLU',
        )

#MLP decoder
traj_dec = dict(
    type = 'MLPDec', # one of 'MLPDec, RNNDec, STDec'
    hidden_size = [1024, 512, 256],
    activation = 'ReLU'
    )
